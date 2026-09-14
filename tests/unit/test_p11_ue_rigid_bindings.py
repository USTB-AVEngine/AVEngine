from __future__ import annotations

from copy import deepcopy
import importlib.util
import json
from pathlib import Path

import pytest

from avengine.capture.qa_plan_adapters import materialize_ue_episode_plan
import avengine.rooms.qa_episode as qa_episode
from avengine.rooms.qa_episode import QAPlanningError, source_declaration
from avengine.runtime_profiles import load_source_asset_runtime_registry


ROOT = Path(__file__).resolve().parents[2]
REGISTRY_PATH = ROOT / "examples/runtime/source_asset_runtime_profiles.json"
SPEAKER_ID = "generated_bookshelf_speaker_compact_shelf_cabinet_black_ash_research_v1"
ANIMAL_ID = "generated_border_collie_black_white_medium_standard_adult_research_v1"


def _load_runner():
    path = ROOT / "tools/rooms/run_spear_apartment_canary.py"
    spec = importlib.util.spec_from_file_location("p11_apartment_runner", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_rigid_source_declaration_uses_static_mesh_resting_pose_and_emitter() -> None:
    registry = load_source_asset_runtime_registry(REGISTRY_PATH)
    declaration = source_declaration(registry, SPEAKER_ID, "source2")

    assert declaration["entity_class"] == "rigid_object"
    assert declaration["motion_model"] == "rigid_static"
    assert declaration["static_mesh_binding"] == "explicit_path"
    assert declaration["static_mesh_object_path"].startswith("/Game/")
    assert declaration["resting_pose"]["attachment_surface"] == "floor"
    assert declaration["emitter_binding"]["semantic_anchor_id"] == "woofer_cone_front_baffle"
    assert declaration["emitter_local_ue_cm"] == pytest.approx(
        [7.6629854739, 1.1091106571, 10.0921750069]
    )
    assert "idle_animation" not in declaration
    assert "walking_animation" not in declaration


def _speaker_registry_copy() -> dict:
    return deepcopy(load_source_asset_runtime_registry(REGISTRY_PATH))


def _speaker_record(registry: dict) -> dict:
    return next(
        item for item in registry["assets"] if item["asset_id"] == SPEAKER_ID
    )


def test_rigid_source_declaration_rejects_missing_resting_pose(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry = _speaker_registry_copy()
    record = _speaker_record(registry)
    record["runtime_backends"]["habitat"].pop("resting_pose")
    monkeypatch.setattr(
        qa_episode,
        "resolve_source_asset_runtime_profile",
        lambda _registry, _asset_id: record,
    )
    monkeypatch.setattr(
        qa_episode,
        "build_asset_emitter_binding",
        lambda _registry, *, source_slot_id, asset_id: {
            "source_slot_id": source_slot_id,
            "asset_id": asset_id,
            "emitter_offset_m": [0.0, 0.0, 0.0],
        },
    )

    with pytest.raises(QAPlanningError, match="interface_not_implemented"):
        source_declaration(registry, SPEAKER_ID, "source2")


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("attachment_surface", "wall"),
        ("attachment_surface", "ceiling"),
        ("base_plane_offset_m", 0.01),
    ],
)
def test_rigid_source_declaration_carries_the_shared_resting_pose_contract(
    field: str, value: object
) -> None:
    registry = _speaker_registry_copy()
    pose = _speaker_record(registry)["runtime_backends"]["habitat"]["resting_pose"]
    pose[field] = value

    declaration = source_declaration(registry, SPEAKER_ID, "source2")
    assert declaration["resting_pose"][field] == value
    assert declaration["static_placement_contract"][field] == value


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("attachment_surface", "roof", "floor, wall or ceiling"),
        ("base_plane_offset_m", float("nan"), "base_plane_offset_m must be finite"),
        ("height_m", -0.01, "height_m must be finite and nonnegative"),
    ],
)
def test_rigid_source_declaration_rejects_invalid_resting_pose(
    monkeypatch: pytest.MonkeyPatch, field: str, value: object, message: str
) -> None:
    registry = _speaker_registry_copy()
    record = _speaker_record(registry)
    record["runtime_backends"]["habitat"]["resting_pose"][field] = value
    monkeypatch.setattr(
        qa_episode,
        "resolve_source_asset_runtime_profile",
        lambda _registry, _asset_id: record,
    )
    monkeypatch.setattr(
        qa_episode,
        "build_asset_emitter_binding",
        lambda _registry, *, source_slot_id, asset_id: {
            "source_slot_id": source_slot_id,
            "asset_id": asset_id,
            "emitter_offset_m": [0.0, 0.0, 0.0],
        },
    )

    with pytest.raises(QAPlanningError, match=f"interface_not_implemented.*{message}"):
        source_declaration(registry, SPEAKER_ID, "source2")


def test_articulated_blueprint_component_binding_does_not_require_explicit_mesh() -> None:
    registry = load_source_asset_runtime_registry(REGISTRY_PATH)
    declaration = source_declaration(registry, ANIMAL_ID, "source2")

    assert declaration["entity_class"] == "articulated_animal"
    assert declaration["skeletal_mesh_binding"] == "blueprint_component"
    assert declaration["skeletal_mesh_path"] is None
    assert declaration["animation_paths_by_action_id"]
    assert declaration["exact_runtime_binding"] is None


def test_ue_materializer_preserves_rigid_static_track_without_animation() -> None:
    registry = load_source_asset_runtime_registry(REGISTRY_PATH)
    actor = source_declaration(registry, SPEAKER_ID, "source1")
    neutral = {
        "plan_coordinates": "renderer_neutral",
        "scene": {},
        "resources": {},
        "visual_plan": {
            "actors": [
                {
                    "actor_id": "source1",
                    "asset_id": SPEAKER_ID,
                    "entity_class": "rigid_object",
                    "emitter_binding": actor["emitter_binding"],
                }
            ],
            "camera": {
                "position_m": [0.0, 1.5, 0.0],
                "basis": {
                    "forward": [1.0, 0.0, 0.0],
                    "right": [0.0, 0.0, 1.0],
                    "up": [0.0, 1.0, 0.0],
                },
                "horizontal_fov_deg": 85.0,
            },
            "frames": [
                {
                    "frame_index": 0,
                    "actor_states": [
                        {
                            "actor_id": "source1",
                            "root_transform": {
                                "translation_m": [1.0, 0.0, 2.0],
                                "rotation_xyzw": [0.0, 0.0, 0.0, 1.0],
                                "scale": [1.0, 1.0, 1.0],
                            },
                        }
                    ],
                    "camera_state": {
                        "position_m": [0.0, 1.5, 0.0],
                        "basis": {
                            "forward": [1.0, 0.0, 0.0],
                            "right": [0.0, 0.0, 1.0],
                            "up": [0.0, 1.0, 0.0],
                        },
                        "horizontal_fov_deg": 85.0,
                    },
                }
            ],
        },
    }

    materialized = materialize_ue_episode_plan(neutral, registry)
    compiled = materialized["visual_plan"]["actors"][0]
    state = materialized["visual_plan"]["frames"][0]["actor_states"][0]
    assert compiled["motion_model"] == "rigid_static"
    assert compiled["static_mesh_object_path"].startswith("/Game/")
    assert state["translation_ue_cm"] == pytest.approx([100.0, 200.0, 0.0])
    assert state["actor_yaw_ue_deg"] == pytest.approx(0.0)
    assert "ue_animation" not in state


class _FakeScaleRoot:
    def GetRelativeScale3D(self, *, as_dict):
        assert as_dict
        return {"X": 1.25, "Y": 1.25, "Z": 1.25}

    def GetActorScale3D(self, **_kwargs):
        raise AssertionError("static readback must use visual-root relative scale")


def test_habitat_root_conversion_preserves_legacy_floor_yaw_sign() -> None:
    import math
    import avengine.optional_backends.spear_apartment as apartment

    pose = apartment.habitat_root_transform_to_ue(
        {
            "translation_m": [1.0, 2.0, 3.0],
            "rotation_xyzw": [0.0, math.sin(math.radians(30.0)), 0.0, math.cos(math.radians(30.0))],
            "scale": [1.0, 1.0, 1.0],
        }
    )
    assert pose["translation_cm"] == pytest.approx([100.0, 300.0, 200.0])
    assert pose["rotation_deg"] == pytest.approx([0.0, 0.0, -60.0])


def test_materializer_keeps_complete_wall_root_pose_and_world_emitter_point() -> None:
    import avengine.optional_backends.spear_apartment as apartment

    registry = load_source_asset_runtime_registry(REGISTRY_PATH)
    wall_id = "generated_smoke_detector_wall_square_white_research_v1"
    root_quaternion = [0.25, 0.30, 0.10, 0.90]
    norm = sum(value * value for value in root_quaternion) ** 0.5
    root_quaternion = [value / norm for value in root_quaternion]
    habitat_rotation = apartment._rotation_matrix_from_xyzw(
        root_quaternion, owner="test root quaternion"
    )
    root_matrix = [
        value
        for row, translation in zip(
            habitat_rotation,
            [1.0, 2.0, 3.0],
            strict=True,
        )
        for value in (*row, translation)
    ] + [0.0, 0.0, 0.0, 1.0]
    neutral = {
        "plan_coordinates": "renderer_neutral",
        "scene": {},
        "resources": {},
        "visual_plan": {
            "actors": [
                {
                    "actor_id": "source1",
                    "asset_id": wall_id,
                    "entity_class": "rigid_object",
                    "static_placement": {
                        "status": "planned",
                        "asset_id": wall_id,
                        "support_identity": {
                            "surface_kind": "wall",
                            "surface_id": "room_wall_01",
                        },
                    },
                }
            ],
            "camera": {
                "position_m": [0.0, 1.5, 0.0],
                "basis": {
                    "forward": [1.0, 0.0, 0.0],
                    "right": [0.0, 0.0, 1.0],
                    "up": [0.0, 1.0, 0.0],
                },
                "horizontal_fov_deg": 85.0,
            },
            "frames": [
                {
                    "frame_index": 0,
                    "actor_states": [
                        {
                            "actor_id": "source1",
                            "root_transform": {
                                "translation_m": [1.0, 2.0, 3.0],
                                "rotation_xyzw": root_quaternion,
                                "matrix_row_major": root_matrix,
                                "scale": [1.0, 1.0, 1.0],
                            },
                            "planned_emitter_m": [1.1, 2.2, 3.3],
                            "emitter_transform": {
                                "position_m": [1.1, 2.2, 3.3],
                            },
                        }
                    ],
                    "camera_state": {
                        "position_m": [0.0, 1.5, 0.0],
                        "basis": {
                            "forward": [1.0, 0.0, 0.0],
                            "right": [0.0, 0.0, 1.0],
                            "up": [0.0, 1.0, 0.0],
                        },
                        "horizontal_fov_deg": 85.0,
                    },
                }
            ],
        },
    }

    materialized = materialize_ue_episode_plan(neutral, registry)
    state = materialized["visual_plan"]["frames"][0]["actor_states"][0]
    full_pose = state["ue_root_transform"]
    assert state["translation_ue_cm"] == pytest.approx([100.0, 300.0, 200.0])
    assert state["planned_emitter_ue_cm"] == pytest.approx([110.0, 330.0, 220.0])
    assert full_pose["rotation_deg"][0] != pytest.approx(0.0)
    assert full_pose["rotation_deg"][1] != pytest.approx(0.0)
    expected = apartment._rotator_quaternion_xyzw(full_pose["rotation_deg"])
    dot = abs(sum(left * right for left, right in zip(expected, full_pose["rotation_xyzw"])))
    assert dot == pytest.approx(1.0, abs=1.0e-6)


def test_materializer_carries_the_measured_static_placement_to_the_renderer() -> None:
    """The seat the room mesh measured is the seat UE is asked to reproduce.

    Planning now seats a wall or ceiling device on the surface a ray finds
    rather than on the plane the catalog fitted, and records the readings that
    justify it. The renderer has to receive that placement, not a re-derivation
    of it, or the pixels and the plan describe different rooms.
    """
    registry = load_source_asset_runtime_registry(REGISTRY_PATH)
    wall_id = "generated_smoke_detector_wall_square_white_research_v1"
    corners = [[x, y, z] for x in (0.9, 1.1) for y in (1.9, 2.1) for z in (2.9, 3.1)]
    placement = {
        "status": "planned",
        "asset_id": wall_id,
        "instance_id": "wall_device",
        "support_identity": {
            "surface_kind": "wall",
            "surface_id": "room_wall_01",
            "surface_normal_offset_m": -0.087,
            "surface_normal_offset_basis": (
                "measured room surface on the mounting axis under this footprint"
            ),
            "contact_point_m": [0.9, 2.0, 3.0],
            "mounting_direction_m": [1.0, 0.0, 0.0],
        },
        "asset_bounds": {
            "world_aabb_min_m": [0.9, 1.9, 2.9],
            "world_aabb_max_m": [1.1, 2.1, 3.1],
            "world_corners_m": corners,
        },
        "placement_checks": {
            "schema": "avengine_static_placement_geometry_checks_v1",
            "status": "pass",
            "failed_checks": [],
            "normal_flipped": True,
            "support_normal_sign": -1.0,
            "mesh_contact_offset": {"status": "measured", "offset_m": -0.087},
        },
    }
    neutral = {
        "plan_coordinates": "renderer_neutral",
        "scene": {},
        "resources": {},
        "visual_plan": {
            "actors": [{
                "actor_id": "source1", "asset_id": wall_id,
                "entity_class": "rigid_object", "static_placement": placement,
            }],
            "camera": {
                "position_m": [0.0, 1.5, 0.0],
                "basis": {"forward": [1.0, 0.0, 0.0], "right": [0.0, 0.0, 1.0],
                          "up": [0.0, 1.0, 0.0]},
                "horizontal_fov_deg": 85.0,
            },
            "frames": [{
                "frame_index": 0,
                "actor_states": [{
                    "actor_id": "source1",
                    "root_transform": {
                        "translation_m": [0.9, 2.0, 3.0],
                        "rotation_xyzw": [0.0, 0.0, 0.0, 1.0],
                        "matrix_row_major": [1.0, 0.0, 0.0, 0.9,
                                             0.0, 1.0, 0.0, 2.0,
                                             0.0, 0.0, 1.0, 3.0,
                                             0.0, 0.0, 0.0, 1.0],
                        "scale": [1.0, 1.0, 1.0],
                    },
                    "emitter_transform": {"position_m": [1.0, 2.05, 3.0]},
                }],
                "camera_state": {
                    "position_m": [0.0, 1.5, 0.0],
                    "basis": {"forward": [1.0, 0.0, 0.0], "right": [0.0, 0.0, 1.0],
                              "up": [0.0, 1.0, 0.0]},
                    "horizontal_fov_deg": 85.0,
                },
            }],
        },
    }

    materialized = materialize_ue_episode_plan(neutral, registry)
    actor = next(row for row in materialized["visual_plan"]["actors"]
                 if row["actor_id"] == "source1")
    carried = actor["static_placement"]
    assert carried["placement_checks"]["status"] == "pass"
    assert carried["placement_checks"]["normal_flipped"] is True
    assert carried["support_identity"]["contact_point_m"] == [0.9, 2.0, 3.0]
    assert carried["asset_bounds"]["world_corners_m"] == corners
    state = materialized["visual_plan"]["frames"][0]["actor_states"][0]
    assert state["translation_ue_cm"] == pytest.approx([90.0, 300.0, 200.0])
    assert state["planned_emitter_ue_cm"] == pytest.approx([100.0, 300.0, 205.0])


class _ResidentialAnchor:
    def __init__(self) -> None:
        self.location = None
        self.rotation = None

    def K2_SetActorLocationAndRotation(self, *, NewLocation, NewRotation, **_kwargs):
        self.location = NewLocation
        self.rotation = NewRotation

    def K2_GetActorLocation(self, *, as_dict):
        assert as_dict
        return self.location

    def K2_GetActorRotation(self, *, as_dict):
        assert as_dict
        return self.rotation


def test_residential_runner_consumes_complete_static_root_pose() -> None:
    path = ROOT / "tools/rooms/run_spear_residential_episode.py"
    spec = importlib.util.spec_from_file_location("d5_residential_runner", path)
    assert spec is not None and spec.loader is not None
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    anchor = _ResidentialAnchor()
    root, animation = runner._apply_residential_actor_state(
        {"motion_model": "rigid_static", "anchor": anchor},
        {
            "action_id": "static",
            "moving": False,
            "translation_ue_cm": [1.0, 2.0, 3.0],
            "actor_yaw_ue_deg": 4.0,
            "ue_root_transform": {
                "translation_cm": [11.0, 22.0, 33.0],
                "rotation_deg": [12.0, 23.0, 34.0],
            },
        },
        4,
    )
    assert root["location_cm"] == [11.0, 22.0, 33.0]
    assert root["rotation_deg"] == [12.0, 23.0, 34.0]
    assert animation["status"] == "not_applicable"


def test_root_readback_gate_checks_complete_static_rotation() -> None:
    import avengine.optional_backends.spear_apartment as apartment

    state = {
        "actor_id": "source1",
        "translation_ue_cm": [11.0, 22.0, 33.0],
        "actor_yaw_ue_deg": 34.0,
        "ue_root_transform": {
            "translation_cm": [11.0, 22.0, 33.0],
            "rotation_deg": [12.0, 23.0, 34.0],
        },
    }
    frames = [{"actor_states": [state]}]
    actor_readbacks = {
        "source1": [
            {
                "frame_index": 0,
                "location_cm": [11.0, 22.0, 33.0],
                "rotation_deg": [12.0, 23.0, 34.0],
            }
        ]
    }
    cameras = [{"frame_index": 0, "location_cm": [0.0, 0.0, 0.0], "rotation_deg": [0.0, 0.0, 0.0]}]
    summary = apartment.summarize_root_readbacks(
        expected_frames=frames,
        actor_readbacks=actor_readbacks,
        camera_readbacks=cameras,
        camera_position_cm=[0.0, 0.0, 0.0],
        camera_yaw_deg=0.0,
        frame_count=1,
    )
    assert summary["source1"]["full_pose_frame_count"] == 1
    assert summary["source1"]["maximum_rotation_error_deg"] == pytest.approx(0.0, abs=1.0e-5)

    drifted = deepcopy(actor_readbacks)
    drifted["source1"][0]["rotation_deg"][0] += 5.0
    with pytest.raises(apartment.SpearApartmentError, match="source1.*drifted"):
        apartment.summarize_root_readbacks(
            expected_frames=frames,
            actor_readbacks=drifted,
            camera_readbacks=cameras,
            camera_position_cm=[0.0, 0.0, 0.0],
            camera_yaw_deg=0.0,
            frame_count=1,
        )


def test_static_scale_readback_uses_visual_root_and_records_observed_value() -> None:
    runner = _load_runner()
    readback = runner._uniform_relative_scale_readback(
        _FakeScaleRoot(), actor_id="source2", requested_scale=1.25
    )

    assert readback == {
        "status": "pass",
        "authority": "visual_root.GetRelativeScale3D",
        "requested_uniform_scale": 1.25,
        "observed_scale_xyz": [1.25, 1.25, 1.25],
        "maximum_absolute_error": 0.0,
    }


class _FakeAnchor:
    def __init__(self) -> None:
        self.location = None
        self.rotation = None

    def K2_SetActorLocationAndRotation(self, *, NewLocation, NewRotation, **_kwargs):
        self.location = NewLocation
        self.rotation = NewRotation

    def K2_GetActorLocation(self, *, as_dict):
        assert as_dict
        return self.location

    def K2_GetActorRotation(self, *, as_dict):
        assert as_dict
        return self.rotation


def test_rigid_apply_state_is_static_and_has_no_animation_phase() -> None:
    runner = _load_runner()
    anchor = _FakeAnchor()
    root, animation = runner._apply_actor_state(
        {"motion_model": "rigid_static", "anchor": anchor},
        {
            "action_id": "static",
            "moving": False,
            "translation_ue_cm": [10.0, 20.0, 30.0],
            "actor_yaw_ue_deg": 17.0,
        },
        4,
    )

    assert root["location_cm"] == [10.0, 20.0, 30.0]
    assert root["rotation_deg"] == [0.0, 0.0, 17.0]
    assert animation == {
        "frame_index": 4,
        "action_id": "static",
        "motion_model": "rigid_static",
        "status": "not_applicable",
    }

    with pytest.raises(RuntimeError, match="animated state"):
        runner._apply_actor_state(
            {"motion_model": "rigid_static", "anchor": anchor},
            {
                "action_id": "walk",
                "moving": True,
                "translation_ue_cm": [10.0, 20.0, 30.0],
                "actor_yaw_ue_deg": 17.0,
            },
            4,
        )


BEAGLE_ID = "rocketbox_dog_beagle_01_m2_v7_world_contact_candidate"


def _beagle_neutral_plan() -> dict:
    return {
        "plan_coordinates": "renderer_neutral",
        "scene": {},
        "resources": {
            "room_package": {
                "visual_scene": {
                    "map_path": "/Game/SPEAR/Scenes/apartment_0000/Maps/apartment_0000"
                }
            }
        },
        "visual_plan": {
            "actors": [
                {
                    "actor_id": "source1",
                    "asset_id": BEAGLE_ID,
                    "entity_class": "articulated_animal",
                }
            ],
            "camera": {
                "position_m": [0.0, 1.5, 0.0],
                "basis": {
                    "forward": [1.0, 0.0, 0.0],
                    "right": [0.0, 0.0, 1.0],
                    "up": [0.0, 1.0, 0.0],
                },
                "horizontal_fov_deg": 85.0,
            },
            "frames": [
                {
                    "frame_index": 0,
                    "actor_states": [
                        {
                            "actor_id": "source1",
                            "action_id": "idle",
                            "action_phase": 0.0,
                            "root_transform": {
                                "translation_m": [1.0, 0.0, 2.0],
                                "rotation_xyzw": [0.0, 0.0, 0.0, 1.0],
                                "scale": [1.0, 1.0, 1.0],
                            },
                        }
                    ],
                    "camera_state": {
                        "position_m": [0.0, 1.5, 0.0],
                        "basis": {
                            "forward": [1.0, 0.0, 0.0],
                            "right": [0.0, 0.0, 1.0],
                            "up": [0.0, 1.0, 0.0],
                        },
                        "horizontal_fov_deg": 85.0,
                    },
                }
            ],
        },
    }


def test_renderer_neutral_materialization_adds_beagle_local_frame_and_keeps_root():
    registry = load_source_asset_runtime_registry(REGISTRY_PATH)
    plan = _beagle_neutral_plan()
    materialized = materialize_ue_episode_plan(plan, registry)
    visual = materialized["visual_plan"]
    marker = visual["ue_neutral_runtime_binding"]
    assert marker["mode"] == "renderer_neutral_asset_frame_v2"
    assert marker["source"] == "materialize_ue_episode_plan"
    assert marker["actor_root_preserved"] is True
    actor = visual["actors"][0]
    correction = actor["ue_neutral_visual_frame_correction"]
    assert correction["rotation_deg"] == pytest.approx([0.0, 0.0, -180.0])
    assert correction["translation_cm"] == [0.0, 0.0, 0.0]
    assert correction["timeline_forward_yaw_deg"] == pytest.approx(0.0)
    assert correction["ue_anatomical_forward_yaw_deg"] == pytest.approx(180.0)
    assert actor["ue_emitter_attachment"]["attachment_type"] == "bone"
    assert actor["ue_emitter_attachment"]["name"] == "beagle-Xtra-Mouth"
    assert visual["frames"][0]["actor_states"][0]["actor_yaw_ue_deg"] == pytest.approx(
        0.0
    )
    assert visual["frames"][0]["actor_states"][0]["anatomical_forward_ue_world"] == pytest.approx(
        [1.0, 0.0, 0.0]
    )
    assert "ue_neutral_runtime_binding" not in plan["visual_plan"]
    assert "ue_neutral_visual_frame_correction" not in plan["visual_plan"]["actors"][0]


def test_legacy_materialization_does_not_enable_neutral_asset_adaptation():
    legacy = {"plan_coordinates": "ue_spear", "visual_plan": {"sentinel": True}}
    materialized = materialize_ue_episode_plan(legacy, {})
    assert materialized == legacy


def test_skeletal_emitter_attachment_uses_declared_socket_and_local_offset():
    from avengine.backends.spear_ue.research_runtime import (
        attach_skeletal_emitter_component,
    )

    class FakeSkeletal:
        uobject = 41

    class FakeEmitter:
        uobject = 42

        def __init__(self):
            self.parent = None
            self.socket = None
            self.location = None

        def SetMobility(self, *, NewMobility):
            assert NewMobility == "Movable"

        def K2_AttachToComponent(self, *, Parent, SocketName, **_kwargs):
            self.parent = Parent
            self.socket = SocketName
            return True

        def GetAttachParent(self, *, as_handle):
            assert as_handle
            return self.parent.uobject

        def K2_SetRelativeLocation(self, *, NewLocation, **_kwargs):
            self.location = NewLocation

    class FakeService:
        def __init__(self):
            self.emitter = FakeEmitter()

        def create_scene_component_for_scene_component(
            self, *, owner, scene_component_name, uclass
        ):
            assert owner is skeletal
            assert scene_component_name == "source1_skeletal_emitter"
            assert uclass == "USceneComponent"
            return self.emitter

    class FakeGame:
        def __init__(self):
            self.unreal_service = FakeService()

    skeletal = FakeSkeletal()
    game = FakeGame()
    emitter = attach_skeletal_emitter_component(
        game,
        actor_id="source1",
        skeletal_component=skeletal,
        attachment={
            "attachment_type": "bone",
            "name": "beagle-Xtra-Mouth",
            "local_offset_cm": [1.0, 2.0, 3.0],
        },
    )
    assert emitter is game.unreal_service.emitter
    assert emitter.parent is skeletal
    assert emitter.socket == "beagle-Xtra-Mouth"
    assert emitter.location == {"X": 1.0, "Y": 2.0, "Z": 3.0}

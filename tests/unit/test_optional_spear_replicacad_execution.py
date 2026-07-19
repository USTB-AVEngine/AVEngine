from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from types import SimpleNamespace
import runpy
import struct
import subprocess
import sys

import pytest

from avengine.optional_backends.spear_replicacad import (
    ArticulatedJointDefault,
    ArticulatedVisual,
    COORDINATE_CONVENTION,
    HabitatTransform,
    PLAN_SCHEMA,
    ReplicaCADImport,
    ReplicaCADScenePlan,
    ReplicaCADSpawn,
    UnrealTransform,
)
from avengine.optional_backends.spear_replicacad_execution import (
    DATASET_LIGHTS_FAITHFUL_PROFILE_ID,
    EDITOR_RESULT_SCHEMA,
    EXECUTION_REQUEST_SCHEMA,
    M5_1_CAPTURE_SCHEMA,
    M5_1_EMITTER_SCHEMA,
    M5_1_FPS,
    M5_1_FRAME_COUNT,
    M5_1_MAP_PATH,
    M5_1_ROOM_ID,
    M5_1_ROUTE_ID,
    M5_1_ROUTE_SCHEMA,
    M5_1_RUNTIME_SCHEMA,
    M5_1_SOURCE_BINDING_SCHEMA,
    M5_1_SOURCE_GATE_SCHEMA,
    M5_1_SOURCE_PROGRAM_SCHEMA,
    ROOM_LOCAL_REVIEW_PROFILE_ID,
    ROUTE_CENTER_FILL_REVIEW_PROFILE_ID,
    ReplicaCADExecutionError,
    apply_replicacad_habitat_lighting_profile,
    apply_replicacad_lighting_profile_to_runtime_plan,
    build_m5_1_replicacad_runtime_plan,
    build_replicacad_execution_request,
    compile_replicacad_lighting_profile,
    configure_replicacad_habitat_lighting_profile,
    load_replicacad_lighting_profiles,
    replicacad_fixed_exposure_profile,
    resolve_replicacad_route_center_fill,
    validate_replicacad_habitat_lighting_readback,
    validate_replicacad_editor_result,
)


REPOSITORY = Path(__file__).resolve().parents[2]


def _apt0_signed_light_records() -> list[dict[str, object]]:
    values = [
        ("0", [-0.91, 2.3, 2.53], 2.9),
        ("1", [-1.4, 2.3, -0.175], 2.9),
        ("2", [-1.4, 2.3, -2.775], 2.9),
        ("3", [6.928, 3.3, 6.849], 11.1),
        ("4", [7.048, 3.3, 2.378], 18.3),
        ("5", [1.65, 2.6, -2.03], -0.13),
        ("6", [6.06, 2.6, -2.67], -0.27),
    ]
    return [
        {
            "light_id": light_id,
            "habitat_position_m": position,
            "ue_position_cm": [100 * position[0], 100 * position[2], 100 * position[1]],
            "dataset_scaled_intensity": intensity,
            "color_rgb": [0.93, 0.98, 1.0],
        }
        for light_id, position, intensity in values
    ]


def _glb(path: Path, *, meshes: int, materials: int = 1, textures: int = 1) -> Path:
    document = {
        "asset": {"version": "2.0"},
        "scenes": [{"nodes": list(range(meshes))}],
        "nodes": [{"mesh": index} for index in range(meshes)],
        "meshes": [{"primitives": []} for _ in range(meshes)],
        "materials": [{} for _ in range(materials)],
        "textures": [{} for _ in range(textures)],
        "images": [{} for _ in range(textures)],
        "buffers": [{"byteLength": 0}],
    }
    encoded = json.dumps(document, separators=(",", ":")).encode("utf-8")
    encoded += b" " * ((-len(encoded)) % 4)
    binary = b""
    payload = b"".join(
        (
            struct.pack("<4sII", b"glTF", 2, 12 + 8 + len(encoded) + 8),
            struct.pack("<II", len(encoded), 0x4E4F534A),
            encoded,
            struct.pack("<II", len(binary), 0x004E4942),
            binary,
        )
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return path


def _spawn(
    spawn_id: str,
    kind: str,
    source_index: int,
    import_id: str,
    *,
    articulated: bool = False,
    visual_paths: tuple[Path, ...] = (),
) -> ReplicaCADSpawn:
    habitat = HabitatTransform((1.0, 2.0, 3.0), (1.0, 0.0, 0.0, 0.0), (1, 1, 1))
    unreal = UnrealTransform((100.0, 300.0, 200.0), (0, 0, 0, 1), (1, 1, 1))
    return ReplicaCADSpawn(
        spawn_id=spawn_id,
        asset_kind=kind,
        source_index=source_index,
        import_id=import_id,
        template_name=import_id,
        habitat_transform=habitat,
        unreal_transform=unreal,
        motion_type="STATIC",
        translation_origin="COM" if articulated else None,
        fixed_base=True if articulated else None,
        joint_defaults=(
            ArticulatedJointDefault("door", "revolute", 0.0, "urdf_zero", False),
        )
        if articulated
        else (),
        articulated_visuals=tuple(
            ArticulatedVisual(
                visual_id=f"link_{index}:visual:000",
                link_name=f"link_{index}",
                mesh_path=path,
                root_from_visual_matrix=(
                    (1.0, 0.0, 0.0, 0.0),
                    (0.0, 1.0, 0.0, 0.0),
                    (0.0, 0.0, 1.0, 0.0),
                    (0.0, 0.0, 0.0, 1.0),
                ),
            )
            for index, path in enumerate(visual_paths)
        ),
    )


@pytest.fixture
def scene_plan(tmp_path: Path) -> ReplicaCADScenePlan:
    dataset = tmp_path / "replicaCAD.scene_dataset_config.json"
    dataset.write_text(
        json.dumps(
            {
                "light_setups": {
                    "default_attributes": {
                        "positive_intensity_scale": 2.5,
                        "negative_intensity_scale": 0.1,
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    scene = tmp_path / "configs/scenes/apt_0.scene_instance.json"
    scene.parent.mkdir(parents=True)
    scene.write_text("{}", encoding="utf-8")
    lighting = tmp_path / "configs/lighting/frl_apartment_stage.lighting_config.json"
    lighting.parent.mkdir(parents=True)
    lighting.write_text(
        json.dumps(
            {
                "lights": {
                    "0": {
                        "type": "point",
                        "position": [1, 2, 3],
                        "intensity": 2.0,
                        "color": [1, 0.9, 0.8],
                    },
                    "1": {
                        "type": "point",
                        "position": [-1, 2, 3],
                        "intensity": -3.0,
                        "color": [1, 1, 1],
                    },
                }
            }
        ),
        encoding="utf-8",
    )
    stage = _glb(tmp_path / "stage.glb", meshes=2)
    chair = _glb(tmp_path / "chair.glb", meshes=1)
    body = _glb(tmp_path / "body.glb", meshes=1)
    door = _glb(tmp_path / "door.glb", meshes=1)
    urdf = tmp_path / "cabinet.urdf"
    urdf.write_text("<robot name='cabinet'/>", encoding="utf-8")
    config = tmp_path / "template.json"
    config.write_text("{}", encoding="utf-8")
    imports = (
        ReplicaCADImport("stage:room", "stage", "room", config, (stage,)),
        ReplicaCADImport("rigid:chair", "rigid", "chair", config, (chair,)),
        ReplicaCADImport(
            "articulated:cabinet",
            "articulated",
            "cabinet",
            config,
            (body, door),
            urdf,
        ),
    )
    spawns = (
        _spawn("stage:000000", "stage", 0, "stage:room"),
        _spawn("rigid:000000", "rigid", 0, "rigid:chair"),
        _spawn("rigid:000001", "rigid", 1, "rigid:chair"),
        _spawn(
            "articulated:000000",
            "articulated",
            0,
            "articulated:cabinet",
            articulated=True,
            visual_paths=(body, door),
        ),
    )
    return ReplicaCADScenePlan(
        schema=PLAN_SCHEMA,
        coordinate_convention=COORDINATE_CONVENTION,
        dataset_config_path=dataset,
        scene_instance_path=scene,
        default_lighting="lighting/frl_apartment_stage",
        imports=imports,
        spawns=spawns,
        source_stage_count=1,
        source_rigid_count=2,
        source_articulated_count=1,
    )


def test_execution_request_separates_import_asset_and_spawn_counts(
    scene_plan: ReplicaCADScenePlan,
) -> None:
    request = build_replicacad_execution_request(scene_plan)

    assert request["counts"] == {
        "logical_import_count": 3,
        "source_glb_count": 4,
        "expected_imported_static_mesh_asset_count": 5,
        "logical_instance_count": 4,
        "logical_instances_by_kind": {"stage": 1, "rigid": 2, "articulated": 1},
        "expected_runtime_mesh_actor_count": 6,
        "articulated_visual_occurrence_count": 2,
    }
    assert len(request["pbr_import"]["source_meshes"]) == 4
    assert request["pbr_import"]["material_override_allowed"] is False
    assert request["spawns"][0]["expected_mesh_actor_count"] == 2
    assert request["spawns"][-1]["expected_mesh_actor_count"] == 2


def test_execution_request_retains_signed_dataset_lights_without_false_ue_claim(
    scene_plan: ReplicaCADScenePlan,
) -> None:
    lighting = build_replicacad_execution_request(scene_plan)["lighting"]

    assert lighting["default_lighting"] == "lighting/frl_apartment_stage"
    assert lighting["lights"][0]["dataset_scaled_intensity"] == 5.0
    assert lighting["lights"][0]["ue_position_cm"] == [100.0, 300.0, 200.0]
    assert lighting["lights"][1]["dataset_scaled_intensity"] == pytest.approx(-0.3)
    assert "not_representable" in lighting["lights"][1]["ue_realization"]


def test_editor_result_must_close_over_all_imports_and_logical_spawns(
    scene_plan: ReplicaCADScenePlan,
) -> None:
    request = build_replicacad_execution_request(scene_plan)
    result = {
        "schema": EDITOR_RESULT_SCHEMA,
        "status": "pass",
        "counts": {
            "imported_source_glb_count": 4,
            "imported_static_mesh_asset_count": 5,
            "logical_instance_count": 4,
            "logical_instances_by_kind": {"stage": 1, "rigid": 2, "articulated": 1},
            "spawned_static_mesh_actor_count": 6,
            "articulated_visual_occurrence_count": 2,
        },
        "logical_spawn_ids": [item["spawn_id"] for item in request["spawns"]],
        "pbr_readback": {
            "material_asset_count": 5,
            "texture_asset_count": 4,
            "material_overrides_applied": False,
        },
    }
    assert validate_replicacad_editor_result(request, result)["status"] == "pass"

    result["counts"]["logical_instance_count"] = 3
    with pytest.raises(ReplicaCADExecutionError, match="logical_instance_count"):
        validate_replicacad_editor_result(request, result)


def test_fixed_exposure_keeps_dataset_light_claim_explicit() -> None:
    profile = replicacad_fixed_exposure_profile(output_gain=1.25)

    assert profile["eye_adaptation"] == "disabled"
    assert profile["fixed_output_gain"] == 1.25
    assert profile["dataset_declared_light_count"] == 7
    assert profile["runtime_positive_point_light_count"] == 5
    assert profile["recorded_negative_fill_count"] == 2
    assert profile["review_light_added"] is False

    with pytest.raises(ReplicaCADExecutionError, match=r"\[0.25,2.0\]"):
        replicacad_fixed_exposure_profile(output_gain=2.01)


def test_room_local_profile_excludes_only_positive_lights_outside_stage_shell() -> None:
    document = load_replicacad_lighting_profiles(
        REPOSITORY / "examples/m6y/replicacad_apt0_lighting_profiles.json"
    )
    request = {"lighting": {"lights": _apt0_signed_light_records()}}

    faithful = compile_replicacad_lighting_profile(
        execution_request=request,
        profile_document=document,
        profile_id=DATASET_LIGHTS_FAITHFUL_PROFILE_ID,
    )
    local = compile_replicacad_lighting_profile(
        execution_request=request,
        profile_document=document,
        profile_id=ROOM_LOCAL_REVIEW_PROFILE_ID,
    )

    assert faithful["active_positive_light_ids"] == ["0", "1", "2", "3", "4"]
    assert faithful["excluded_positive_light_ids"] == []
    assert faithful["ue_intensity_scale"] == 1.0
    assert faithful["habitat_intensity_scale"] == 1.0
    assert faithful["ue_source_intensities_scaled"] is False
    assert faithful["habitat_source_intensities_scaled"] is False
    assert local["active_positive_light_ids"] == ["0", "1", "2"]
    assert local["excluded_positive_light_ids"] == ["3", "4"]
    assert local["ue_intensity_scale"] == 2.0
    assert local["habitat_intensity_scale"] == 4.0
    assert local["ue_source_intensities_scaled"] is True
    assert local["habitat_source_intensities_scaled"] is True
    assert local["habitat_usage"] == "research_comparison_only"
    assert local["habitat_maintained_default"] == "no_lights_plus_hbao"
    assert local["stage_shadow_mode"] == "source_import_default"
    assert local["source_lights_moved"] is False
    assert local["review_light_added"] is False
    outside = {
        item["light_id"]
        for item in local["source_positive_lights"]
        if not item["inside_stage_shell_aabb"]
    }
    assert outside == {"3", "4"}


def test_room_local_profile_updates_runtime_counts_without_changing_authority() -> None:
    inputs = _m5_1_runtime_inputs()
    base = build_m5_1_replicacad_runtime_plan(**inputs)
    document = load_replicacad_lighting_profiles(
        REPOSITORY / "examples/m6y/replicacad_apt0_lighting_profiles.json"
    )
    profile = compile_replicacad_lighting_profile(
        execution_request=inputs["execution_request"],
        profile_document=document,
        profile_id=ROOM_LOCAL_REVIEW_PROFILE_ID,
    )

    plan = apply_replicacad_lighting_profile_to_runtime_plan(base, profile)

    assert plan["authority"] == base["authority"]
    assert plan["scene"]["dataset_point_light_actor_count"] == 5
    assert plan["scene"]["runtime_positive_point_light_count"] == 3
    assert plan["scene"]["stage_static_mesh_actor_count"] == 20
    assert plan["exposure_and_lighting"]["lighting_profile_id"] == (
        ROOM_LOCAL_REVIEW_PROFILE_ID
    )
    assert plan["exposure_and_lighting"]["excluded_positive_light_ids"] == ["3", "4"]
    assert plan["exposure_and_lighting"]["ue_intensity_scale"] == 2.0
    assert plan["exposure_and_lighting"]["habitat_intensity_scale"] == 4.0
    assert plan["exposure_and_lighting"]["source_intensities_scaled"] is True


def _route_manifest_for_generated_fill() -> dict[str, object]:
    return {
        "schema": M5_1_ROUTE_SCHEMA,
        "routes": {
            "human0": {
                "start_m": [1.0, 0.1193729192, 5.4],
                "end_m": [1.0, 0.1193729192, 6.6],
            },
            "dog0": {
                "start_m": [3.0, 0.1193729192, 5.4],
                "end_m": [3.0, 0.1193729192, 6.6],
            },
        },
    }


def test_route_center_fill_resolves_from_actor_routes_inside_stage_shell() -> None:
    document = load_replicacad_lighting_profiles(
        REPOSITORY / "examples/m6y/replicacad_apt0_lighting_profiles.json"
    )
    profile = compile_replicacad_lighting_profile(
        execution_request={"lighting": {"lights": _apt0_signed_light_records()}},
        profile_document=document,
        profile_id=ROUTE_CENTER_FILL_REVIEW_PROFILE_ID,
    )

    assert profile["review_light_added"] is True
    assert profile["generated_interior_fill"]["resolved"] is False
    resolved = resolve_replicacad_route_center_fill(
        profile, _route_manifest_for_generated_fill()
    )
    fill = resolved["generated_interior_fill"]

    assert fill["resolved"] is True
    assert fill["route_endpoint_count"] == 4
    assert fill["habitat_position_m"] == pytest.approx(
        [2.0, 3.0445577901000036 - 0.45, 6.0]
    )
    assert fill["ue_position_cm"] == pytest.approx(
        [200.0, 600.0, (3.0445577901000036 - 0.45) * 100.0]
    )


def test_route_center_fill_updates_runtime_counts_without_changing_authority() -> None:
    inputs = _m5_1_runtime_inputs()
    base = build_m5_1_replicacad_runtime_plan(**inputs)
    document = load_replicacad_lighting_profiles(
        REPOSITORY / "examples/m6y/replicacad_apt0_lighting_profiles.json"
    )
    profile = compile_replicacad_lighting_profile(
        execution_request=inputs["execution_request"],
        profile_document=document,
        profile_id=ROUTE_CENTER_FILL_REVIEW_PROFILE_ID,
    )
    profile = resolve_replicacad_route_center_fill(
        profile, _route_manifest_for_generated_fill()
    )

    plan = apply_replicacad_lighting_profile_to_runtime_plan(base, profile)

    assert plan["authority"] == base["authority"]
    assert plan["scene"]["runtime_active_dataset_point_light_count"] == 3
    assert plan["scene"]["generated_review_point_light_count"] == 1
    assert plan["scene"]["runtime_positive_point_light_count"] == 4
    assert plan["scene"]["review_light_added"] is True
    exposure = plan["exposure_and_lighting"]
    assert exposure["review_light_added"] is True
    assert "not dataset-authored" in exposure["claim_boundary"]


@dataclass(frozen=True)
class _FakeLightInfo:
    vector: tuple[float, float, float, float]
    color: tuple[float, float, float]
    model: str


class _FakeHabitatSimulator:
    def __init__(self, configuration: SimpleNamespace) -> None:
        self.config = configuration
        self.current: list[_FakeLightInfo] = []
        self.setups: dict[str, list[_FakeLightInfo]] = {}

    def set_light_setup(self, setup: list[_FakeLightInfo], key: str) -> None:
        copied = list(setup)
        self.setups[str(key)] = copied
        if str(key) == "default-light-key":
            self.current = copied

    def get_current_light_setup(self) -> list[_FakeLightInfo]:
        return list(self.current)

    def get_light_setup(self, key: str) -> list[_FakeLightInfo]:
        return list(self.setups[str(key)])


def test_habitat_room_local_profile_scales_same_three_source_lights() -> None:
    document = load_replicacad_lighting_profiles(
        REPOSITORY / "examples/m6y/replicacad_apt0_lighting_profiles.json"
    )
    profile = compile_replicacad_lighting_profile(
        execution_request={"lighting": {"lights": _apt0_signed_light_records()}},
        profile_document=document,
        profile_id=ROOM_LOCAL_REVIEW_PROFILE_ID,
    )
    fake_habitat = SimpleNamespace(
        gfx=SimpleNamespace(
            DEFAULT_LIGHTING_KEY="default-light-key",
            LightInfo=_FakeLightInfo,
            LightPositionModel=SimpleNamespace(Global="global"),
        )
    )
    configuration = SimpleNamespace(
        sim_cfg=SimpleNamespace(
            scene_light_setup="lighting/frl_apartment_stage",
            override_scene_light_defaults=False,
        )
    )
    configured = configure_replicacad_habitat_lighting_profile(
        configuration=configuration,
        habitat_sim=fake_habitat,
        lighting_profile=profile,
    )
    simulator = _FakeHabitatSimulator(configuration)
    applied = apply_replicacad_habitat_lighting_profile(
        simulator=simulator,
        lighting_profile=profile,
        habitat_sim=fake_habitat,
        actor_light_setup_key="actor-key",
    )
    readback = validate_replicacad_habitat_lighting_readback(
        simulator=simulator,
        lighting_profile=profile,
        habitat_sim=fake_habitat,
        actor_light_setup_key="actor-key",
    )

    assert configured["override_scene_light_defaults"] is True
    assert applied["active_light_ids"] == ["0", "1", "2"]
    assert applied["habitat_intensity_scale"] == 4.0
    assert applied["source_intensities_scaled"] is True
    assert applied["habitat_usage"] == "research_comparison_only"
    assert applied["habitat_maintained_default"] == "no_lights_plus_hbao"
    assert simulator.current[0].color == pytest.approx(
        (0.93 * 2.9 * 4.0, 0.98 * 2.9 * 4.0, 1.0 * 2.9 * 4.0)
    )
    assert readback["current_matches_profile"] is True
    assert readback["actor_setup_matches_profile"] is True


def test_habitat_route_center_profile_adds_one_explicit_fill() -> None:
    document = load_replicacad_lighting_profiles(
        REPOSITORY / "examples/m6y/replicacad_apt0_lighting_profiles.json"
    )
    profile = compile_replicacad_lighting_profile(
        execution_request={"lighting": {"lights": _apt0_signed_light_records()}},
        profile_document=document,
        profile_id=ROUTE_CENTER_FILL_REVIEW_PROFILE_ID,
    )
    profile = resolve_replicacad_route_center_fill(
        profile, _route_manifest_for_generated_fill()
    )
    fake_habitat = SimpleNamespace(
        gfx=SimpleNamespace(
            DEFAULT_LIGHTING_KEY="default-light-key",
            LightInfo=_FakeLightInfo,
            LightPositionModel=SimpleNamespace(Global="global"),
        )
    )
    configuration = SimpleNamespace(
        sim_cfg=SimpleNamespace(
            scene_light_setup="lighting/frl_apartment_stage",
            override_scene_light_defaults=False,
        )
    )
    configure_replicacad_habitat_lighting_profile(
        configuration=configuration,
        habitat_sim=fake_habitat,
        lighting_profile=profile,
    )
    simulator = _FakeHabitatSimulator(configuration)

    applied = apply_replicacad_habitat_lighting_profile(
        simulator=simulator,
        lighting_profile=profile,
        habitat_sim=fake_habitat,
        actor_light_setup_key="actor-key",
    )

    assert applied["active_light_count"] == 4
    assert applied["review_light_added"] is True
    fill = profile["generated_interior_fill"]
    assert simulator.current[-1].vector == pytest.approx(
        (*fill["habitat_position_m"], 1.0)
    )
    assert simulator.current[-1].color == pytest.approx((3.0, 3.0, 3.0))
    assert applied["lights"][-1]["generated_review_light"] is True


def _m5_1_runtime_inputs() -> dict[str, object]:
    spawn_ids = [f"spawn:{index:06d}" for index in range(120)]
    execution_request = {
        "schema": EXECUTION_REQUEST_SCHEMA,
        "counts": {
            "logical_import_count": 87,
            "source_glb_count": 101,
            "expected_imported_static_mesh_asset_count": 127,
            "logical_instance_count": 120,
            "logical_instances_by_kind": {
                "stage": 1,
                "rigid": 113,
                "articulated": 6,
            },
            "expected_runtime_mesh_actor_count": 171,
            "articulated_visual_occurrence_count": 31,
        },
        "lighting": {
            "default_lighting": "lighting/frl_apartment_stage",
            "lights": _apt0_signed_light_records(),
        },
        "spawns": [{"spawn_id": value} for value in spawn_ids],
    }
    editor_result = {
        "schema": EDITOR_RESULT_SCHEMA,
        "status": "pass",
        "counts": {
            "imported_source_glb_count": 101,
            "imported_static_mesh_asset_count": 127,
            "logical_instance_count": 120,
            "logical_instances_by_kind": {
                "stage": 1,
                "rigid": 113,
                "articulated": 6,
            },
            "spawned_static_mesh_actor_count": 171,
            "articulated_visual_occurrence_count": 31,
        },
        "logical_spawn_ids": spawn_ids,
        "pbr_readback": {
            "material_asset_count": 111,
            "texture_asset_count": 106,
            "material_overrides_applied": False,
        },
        "map": {"object_path": M5_1_MAP_PATH},
        "lighting": {"positive_dataset_light_count": 5},
    }
    editor_reload_result = json.loads(json.dumps(editor_result))
    editor_reload_result["map"]["reloaded"] = True
    editor_reload_result["reload_verification"] = "pass"

    floor_y = 0.1193729192
    routes = {
        "human0": {"start_m": [1.0, floor_y, 5.4], "end_m": [1.0, floor_y, 6.6]},
        "dog0": {"start_m": [3.0, floor_y, 5.4], "end_m": [3.0, floor_y, 6.6]},
    }
    route_manifest = {
        "schema": M5_1_ROUTE_SCHEMA,
        "route_id": M5_1_ROUTE_ID,
        "room_id": M5_1_ROOM_ID,
        "frame_count": M5_1_FRAME_COUNT,
        "frame_rate_hz": M5_1_FPS,
        "center_navigation_semantics": "actor_root_center_only",
        "routes": routes,
    }
    frame_readback = []
    positions: dict[str, list[list[float]]] = {"human0": [], "dog0": []}
    for frame_index in range(M5_1_FRAME_COUNT):
        actor_positions = {}
        for actor_id, route in routes.items():
            position = [
                float(start)
                + (float(end) - float(start)) * frame_index / (M5_1_FRAME_COUNT - 1)
                for start, end in zip(route["start_m"], route["end_m"])
            ]
            positions[actor_id].append(position)
            actor_positions[actor_id] = position
        frame_readback.append(
            {
                "frame_index": frame_index,
                "pts_ticks": frame_index * 3200,
                "human": {
                    "actor_root_position_m": actor_positions["human0"],
                    "action_sample_index": frame_index % 16,
                },
                "beagle": {
                    "actor_root_position_m": actor_positions["dog0"],
                    "action_sample_index": (frame_index % 45) % 25,
                },
            }
        )
    capture_evidence = {
        "schema": M5_1_CAPTURE_SCHEMA,
        "status": "pass",
        "room_id": M5_1_ROOM_ID,
        "route_id": M5_1_ROUTE_ID,
        "frame_count": M5_1_FRAME_COUNT,
        "frame_rate_hz": M5_1_FPS,
        "time_base_hz": 48_000,
        "qualification_claim": False,
        "research_only": True,
        "camera": {
            "position_m": [2.6, 1.47, 3.4],
            "rotation_xyzw": [0, 1, 0, 0],
            "horizontal_fov_deg": 90,
        },
    }
    source_center_gate = {
        "schema": M5_1_SOURCE_GATE_SCHEMA,
        "status": "pass",
        "semantics": "source_center_only",
        "pathfinder_snapshot_match": True,
        "full_body_collision_claim": False,
        "failed_source_frame_indices": {},
        "sources": {
            source_id: {
                "status": "pass",
                "frame_count": M5_1_FRAME_COUNT,
                "failed_frame_indices": [],
                "minimum_navmesh_clearance_m": 0.2,
                "minimum_loaded_rigid_clearance_m": 0.3,
                "minimum_blocking_loaded_rigid_clearance_m": 0.3,
            }
            for source_id in ("source0", "source1")
        },
    }
    source_program = {
        "schema": M5_1_SOURCE_PROGRAM_SCHEMA,
        "room_family": "replicacad",
        "sources": [
            {"source_id": "source0", "event_windows": [{"action": "speech"}]},
            {"source_id": "source1", "event_windows": [{"action": "bark"}]},
        ],
    }
    emitter_trajectories = {
        "schema": M5_1_EMITTER_SCHEMA,
        "source_ids": ["source0", "source1"],
        "sources": {
            "source0": {
                "source_id": "source0",
                "actor_id": "human0",
                "emitter_anchor_id": "mouth",
                "frame_count": M5_1_FRAME_COUNT,
                "positions_m": positions["human0"],
                "trajectory_content_sha256": "0" * 64,
            },
            "source1": {
                "source_id": "source1",
                "actor_id": "dog0",
                "emitter_anchor_id": "mouth",
                "frame_count": M5_1_FRAME_COUNT,
                "positions_m": positions["dog0"],
                "trajectory_content_sha256": "1" * 64,
            },
        },
    }
    source_actor_bindings = {
        "schema": M5_1_SOURCE_BINDING_SCHEMA,
        "room_id": M5_1_ROOM_ID,
        "route_id": M5_1_ROUTE_ID,
        "source_ids": ["source0", "source1"],
        "bindings": {
            "source0": {"actor_id": "human0"},
            "source1": {"actor_id": "dog0"},
        },
    }
    return {
        "route_manifest": route_manifest,
        "capture_evidence": capture_evidence,
        "frame_readback": frame_readback,
        "source_center_gate": source_center_gate,
        "source_program": source_program,
        "emitter_trajectories": emitter_trajectories,
        "source_actor_bindings": source_actor_bindings,
        "execution_request": execution_request,
        "editor_import_result": editor_result,
        "editor_reload_result": editor_reload_result,
    }


def test_m5_1_runtime_plan_closes_authority_clock_source_gate_and_editor_result() -> (
    None
):
    plan = build_m5_1_replicacad_runtime_plan(**_m5_1_runtime_inputs())

    assert plan["schema"] == M5_1_RUNTIME_SCHEMA
    assert plan["authority"]["backend_may_replan"] is False
    assert plan["clock"]["frame_count"] == 270
    assert plan["clock"]["duration_ticks"] == 864_000
    assert plan["clock"]["timeline_v2_applicable"] is False
    assert len(plan["frames"]) == 270
    assert plan["frames"][-1]["pts_ticks"] == 860_800
    assert plan["source_logic"]["source_center_gate"]["status"] == "pass"
    assert (
        plan["source_logic"]["source_center_gate"]["full_body_collision_claim"] is False
    )
    assert plan["scene"]["static_mesh_actor_count"] == 171
    assert plan["scene"]["pbr_material_count"] == 111
    assert plan["scene"]["runtime_positive_point_light_count"] == 5
    assert plan["route_characterization"]["normal_speed_issue_resolved"] is False


def test_m5_1_runtime_plan_rejects_broken_editor_and_source_center_closure() -> None:
    inputs = _m5_1_runtime_inputs()
    inputs["editor_reload_result"]["counts"]["spawned_static_mesh_actor_count"] = 170
    with pytest.raises(
        ReplicaCADExecutionError, match="spawned_static_mesh_actor_count"
    ):
        build_m5_1_replicacad_runtime_plan(**inputs)

    inputs = _m5_1_runtime_inputs()
    inputs["source_center_gate"]["status"] = "fail"
    with pytest.raises(ReplicaCADExecutionError, match="source-center gate"):
        build_m5_1_replicacad_runtime_plan(**inputs)


def test_replicacad_runner_dry_run_compiles_without_unreal(tmp_path: Path) -> None:
    inputs = _m5_1_runtime_inputs()
    paths = {}
    for name, value in inputs.items():
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps(value), encoding="utf-8")
        paths[name] = path
    output = tmp_path / "dry_run"
    command = [
        sys.executable,
        str(REPOSITORY / "tools/m6y/run_spear_replicacad_canary.py"),
        "--unreal-editor",
        str(tmp_path / "not-needed-for-dry-run"),
        "--ue-project",
        str(tmp_path / "not-needed-for-dry-run.uproject"),
        "--output-dir",
        str(output),
        "--dry-run",
    ]
    for name in inputs:
        command.extend(("--" + name.replace("_", "-"), str(paths[name])))
    completed = subprocess.run(command, capture_output=True, text=True, check=False)

    assert completed.returncode == 0, completed.stderr
    assert "SPEAR_REPLICACAD_DRY_RUN_OK" in completed.stdout
    plan = json.loads((output / "execution_plan.json").read_text(encoding="utf-8"))
    assert plan["schema"] == M5_1_RUNTIME_SCHEMA
    assert plan["authority"]["backend_may_replan"] is False


def test_environment_probe_has_no_private_data_engine_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("UNREAL_ENGINE_DIR", raising=False)
    monkeypatch.delenv("UE_ENGINE_DIR", raising=False)
    namespace = runpy.run_path(
        str(REPOSITORY / "tools/m6y/probe_spear_replicacad_environment.py")
    )

    candidates = namespace["_engine_candidates"](None)
    assert candidates == [Path("/opt/UnrealEngine")]
    assert all(not str(path).startswith("/data/") for path in candidates)

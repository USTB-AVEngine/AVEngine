from __future__ import annotations

import json
from pathlib import Path
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
    ReplicaCADExecutionError,
    build_m5_1_replicacad_runtime_plan,
    build_replicacad_execution_request,
    replicacad_fixed_exposure_profile,
    validate_replicacad_editor_result,
)


REPOSITORY = Path(__file__).resolve().parents[2]


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
        "lighting": {"default_lighting": "lighting/frl_apartment_stage"},
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
                + (float(end) - float(start))
                * frame_index
                / (M5_1_FRAME_COUNT - 1)
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


def test_m5_1_runtime_plan_closes_authority_clock_source_gate_and_editor_result() -> None:
    plan = build_m5_1_replicacad_runtime_plan(**_m5_1_runtime_inputs())

    assert plan["schema"] == M5_1_RUNTIME_SCHEMA
    assert plan["authority"]["backend_may_replan"] is False
    assert plan["clock"]["frame_count"] == 270
    assert plan["clock"]["duration_ticks"] == 864_000
    assert plan["clock"]["timeline_v2_applicable"] is False
    assert len(plan["frames"]) == 270
    assert plan["frames"][-1]["pts_ticks"] == 860_800
    assert plan["source_logic"]["source_center_gate"]["status"] == "pass"
    assert plan["source_logic"]["source_center_gate"]["full_body_collision_claim"] is False
    assert plan["scene"]["static_mesh_actor_count"] == 171
    assert plan["scene"]["pbr_material_count"] == 111
    assert plan["scene"]["runtime_positive_point_light_count"] == 5
    assert plan["route_characterization"]["normal_speed_issue_resolved"] is False


def test_m5_1_runtime_plan_rejects_broken_editor_and_source_center_closure() -> None:
    inputs = _m5_1_runtime_inputs()
    inputs["editor_reload_result"]["counts"]["spawned_static_mesh_actor_count"] = 170
    with pytest.raises(ReplicaCADExecutionError, match="spawned_static_mesh_actor_count"):
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
        "--unreal-editor", str(tmp_path / "not-needed-for-dry-run"),
        "--ue-project", str(tmp_path / "not-needed-for-dry-run.uproject"),
        "--output-dir", str(output),
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


def test_environment_probe_has_no_private_data_engine_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("UNREAL_ENGINE_DIR", raising=False)
    monkeypatch.delenv("UE_ENGINE_DIR", raising=False)
    namespace = runpy.run_path(
        str(REPOSITORY / "tools/m6y/probe_spear_replicacad_environment.py")
    )

    candidates = namespace["_engine_candidates"](None)
    assert candidates == [Path("/opt/UnrealEngine")]
    assert all(not str(path).startswith("/data/") for path in candidates)

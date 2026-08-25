from __future__ import annotations

from contextlib import nullcontext
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess

import numpy as np
import pytest

from avengine.contracts.json_io import canonical_json_sha256
import avengine.optional_backends.spear_apartment as apartment
from avengine.capture.orientation import habitat_yaw_degrees_from_xyzw
from avengine.sensor_rig_trajectory import materialize_sensor_rig_trajectory


_RUNNER_SPEC = importlib.util.spec_from_file_location(
    "run_spear_apartment_canary",
    Path(__file__).resolve().parents[2] / "tools/rooms/run_spear_apartment_canary.py",
)
assert _RUNNER_SPEC is not None and _RUNNER_SPEC.loader is not None
_RUNNER = importlib.util.module_from_spec(_RUNNER_SPEC)
_RUNNER_SPEC.loader.exec_module(_RUNNER)

CURRENT_GENERATED_DOG_ASSET_ID = apartment.BORDER_COLLIE_ASSET_ID
CURRENT_GENERATED_CAT_ASSET_ID = apartment.CAT_ASSET_ID


def _plan(scenario_id: str = "S3") -> dict:
    actors = [
        {
            "actor_id": "dog0",
            "asset_id": apartment.BEAGLE_ASSET_ID,
            "blueprint_class_path": "dog_bp",
            "idle_animation": "dog_idle",
            "walking_animation": "dog_walk",
        },
        {
            "actor_id": "human0",
            "asset_id": apartment.HUMAN_ASSET_ID,
            "blueprint_class_path": "human_bp",
            "idle_animation": "human_idle",
            "walking_animation": "human_walk",
        },
    ]
    frames = []
    for frame_index in range(apartment.FRAME_COUNT):
        frames.append(
            {
                "frame_index": frame_index,
                "pts_ticks": frame_index * 3_200,
                "actor_states": [
                    {
                        "actor_id": "dog0",
                        "translation_ue_cm": [1.0, 2.0, 27.1],
                        "actor_yaw_ue_deg": -90.0,
                        "anatomical_forward_ue_world": [0.0, -1.0, 0.0],
                    },
                    {
                        "actor_id": "human0",
                        "translation_ue_cm": [3.0 + frame_index, 4.0, 27.1],
                        "actor_yaw_ue_deg": -145.0,
                        "anatomical_forward_ue_world": [
                            0.573576436,
                            -0.819152044,
                            0.0,
                        ],
                    },
                ],
            }
        )
    return {
        "schema": apartment.PLAN_SCHEMA,
        "backend_role": apartment.BACKEND_ROLE,
        "authority": {"backend_may_replan": False},
        "room": {
            "source_scene_provenance": {
                "provider": "SPEAR_Unreal",
                "scene_id": "apartment_0000",
            }
        },
        "camera": {
            "ue_position_cm": [-70.0, 65.0, 147.1],
            "ue_yaw_deg": -145.0,
            "horizontal_fov_deg": 105.0,
        },
        "render": {
            "frame_count": 75,
            "fps_num": 15,
            "fps_den": 1,
            "ticks_per_frame": 3_200,
        },
        "actors": actors,
        "source_logic": {"scenario_id": scenario_id},
        "frames": frames,
    }


def _dynamic_camera_plan() -> dict:
    plan = _plan()
    trajectory = materialize_sensor_rig_trajectory(
        trajectory_id="apartment_dynamic_camera_test_v1",
        program={
            "kind": "WAYPOINT_ROUTE",
            "waypoints": [
                {
                    "frame_index": 0,
                    "position_m": [-0.7, 1.471, 0.65],
                    "yaw_deg": 55.0,
                },
                {
                    "frame_index": 74,
                    "position_m": [0.8, 1.471, -1.25],
                    "yaw_deg": -35.0,
                },
            ],
            "interpolation": "LINEAR_POSITION_SHORTEST_YAW",
        },
    )
    for frame, trajectory_frame in zip(plan["frames"], trajectory["frames"]):
        world_from_rig = deepcopy(trajectory_frame["world_from_rig"])
        position = world_from_rig["translation_m"]
        yaw = habitat_yaw_degrees_from_xyzw(
            world_from_rig["rotation_xyzw"]
        )
        frame["camera_state"] = {
            "frame_index": trajectory_frame["frame_index"],
            "pts_ticks": trajectory_frame["pts_ticks"],
            "world_from_rig": world_from_rig,
            "habitat_position_m": list(position),
            "habitat_yaw_deg": yaw,
            "ue_position_cm": list(
                apartment.habitat_point_to_apartment_ue_cm(position)
            ),
            "ue_yaw_deg": apartment.camera_ue_yaw_degrees(yaw),
            "pose_hash": trajectory_frame["pose_hash"],
        }
    first = plan["frames"][0]["camera_state"]
    plan["camera"]["ue_position_cm"] = deepcopy(first["ue_position_cm"])
    plan["camera"]["ue_yaw_deg"] = first["ue_yaw_deg"]
    return plan


def _make_input_tree(root: Path, scenario_id: str = "S3") -> dict[str, Path]:
    scenario_directory, variant_id = apartment.SCENARIO_DIRECTORIES[scenario_id]
    metadata = (
        root / "scenarios" / scenario_directory / "variants" / variant_id / "metadata"
    )
    videos = metadata.parent / "videos"
    metadata.mkdir(parents=True)
    videos.mkdir()
    values = {
        "timeline": metadata / "timeline.json",
        "source_manifest": metadata / "source_manifest.json",
        "flags": metadata / "flags.json",
        "room_capsule": root / "inputs/fixed_apartment_config/room_capsule.json",
        "qualification": root / "room/qualification.json",
        "authoritative_clean_binaural": videos / "clean_binaural.mp4",
        "authoritative_diagnostic_topdown": videos / "diagnostic_topdown_binaural.mp4",
    }
    for path in values.values():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"{}\n")
    return values


def _make_motion_pilot_input_tree(
    root: Path, scenario_id: str = "P0"
) -> dict[str, Path]:
    episode_directory = apartment.MOTION_PILOT_DIRECTORIES[scenario_id]
    metadata = root / "episodes" / episode_directory / "metadata"
    videos = metadata.parent / "videos"
    metadata.mkdir(parents=True)
    videos.mkdir()
    values = {
        "timeline": metadata / "timeline.json",
        "source_manifest": metadata / "source_manifest.json",
        "flags": metadata / "flags.json",
        "room_capsule": root / "room/room_capsule.json",
        "qualification": root / "room/qualification.json",
        "authoritative_clean_binaural": videos / "clean_binaural.mp4",
        "authoritative_diagnostic_topdown": videos / "diagnostic_topdown_binaural.mp4",
    }
    for path in values.values():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"{}\n")
    return values


def _make_asset_bound_input_tree(
    root: Path, episode_id: str = "episode_0001"
) -> dict[str, Path]:
    metadata = root / "episodes" / episode_id / "metadata"
    videos = metadata.parent / "videos"
    metadata.mkdir(parents=True)
    videos.mkdir()
    values = {
        "timeline": metadata / "timeline.json",
        "source_manifest": metadata / "source_manifest.json",
        "flags": metadata / "flags.json",
        "batch_binding": metadata / "batch_binding.json",
        "sensor_rig_trajectory": metadata / "sensor_rig_trajectory.json",
        "room_capsule": root / "room/room_capsule.json",
        "qualification": root / "room/qualification.json",
        "authoritative_clean_binaural": videos / "clean_binaural.mp4",
        "authoritative_diagnostic_topdown": videos
        / "diagnostic_topdown_binaural.mp4",
    }
    for path in values.values():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"{}\n")
    return values


def _acoustic_selection_binding(selection_mode: str) -> dict:
    binding = {
        "schema": apartment.ACOUSTIC_SELECTION_BINDING_SCHEMA,
        "selection_mode": selection_mode,
        "registry_selection_applied": selection_mode.startswith("registry"),
        "room_ref": (
            deepcopy(apartment.DEFAULT_ROOM_RUNTIME_PROFILE["room_ref"])
            if selection_mode.startswith("registry")
            else None
        ),
        "profile_ref": (
            {"profile_id": "test_acoustic_profile", "revision": "v1"}
            if selection_mode.startswith("registry")
            else None
        ),
        "binding_id": (
            "test_room_to_acoustic_profile_v1"
            if selection_mode.startswith("registry")
            else None
        ),
        "registry_selection_content_sha256": "1" * 64,
        "effective_selection_content_sha256": "2" * 64,
        "acoustic_package_manifest_sha256": "3" * 64,
        "simulation_request_sha256": "4" * 64,
        "input_receipt_sha256": "5" * 64,
        "binding_content_sha256": None,
    }
    if selection_mode != "explicit_legacy_unbound":
        binding["binding_content_sha256"] = canonical_json_sha256(
            {
                key: value
                for key, value in binding.items()
                if key != "binding_content_sha256"
            }
        )
    return binding


def _write_asset_bound_identity_manifest(
    root: Path,
    *,
    selection_mode: str = "registry",
    visual_room_ref: dict | None = None,
    binding_room_ref: dict | None = None,
) -> dict:
    visual = deepcopy(
        visual_room_ref or apartment.DEFAULT_ROOM_RUNTIME_PROFILE["room_ref"]
    )
    binding = _acoustic_selection_binding(selection_mode)
    if binding_room_ref is not None:
        binding["room_ref"] = deepcopy(binding_room_ref)
        binding["binding_content_sha256"] = canonical_json_sha256(
            {
                key: value
                for key, value in binding.items()
                if key != "binding_content_sha256"
            }
        )
    acoustic = deepcopy(binding["room_ref"])
    manifest = {
        "schema": apartment.ASSET_BOUND_BUNDLE_SCHEMA,
        "status": "pass",
        "episode_count": 1,
        "episode_ids": ["episode_0001"],
        "visual_room_ref": visual,
        "acoustic_selection_binding": binding,
        "acoustic_visual_room_alignment": {
            "status": "pass" if acoustic is not None else "not_verified",
            "compatibility": (
                None
                if acoustic is not None
                else "legacy_acoustic_selection_without_room_ref"
            ),
            "visual_room_ref": visual,
            "acoustic_room_ref": acoustic,
        },
    }
    (root / "manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    return manifest


def test_asset_bound_runtime_identity_closes_registry_room_and_binding(
    tmp_path: Path,
) -> None:
    manifest = _write_asset_bound_identity_manifest(tmp_path)

    identity = apartment.asset_bound_bundle_acoustic_visual_identity(tmp_path)

    expected_room_ref = apartment.DEFAULT_ROOM_RUNTIME_PROFILE["room_ref"]
    assert identity["status"] == "pass"
    assert identity["verification_status"] == "verified"
    assert identity["visual_room_ref"] == expected_room_ref
    assert identity["acoustic_room_ref"] == expected_room_ref
    assert identity["runtime_room_ref"] == expected_room_ref
    assert identity["runtime_map_id"] == "apartment_0000"
    assert identity["acoustic_selection_binding_sha256"] == manifest[
        "acoustic_selection_binding"
    ]["binding_content_sha256"]


@pytest.mark.parametrize(
    "selection_mode", ["explicit_legacy", "explicit_legacy_unbound"]
)
def test_legacy_asset_bound_acoustic_identity_stays_not_verified(
    tmp_path: Path, selection_mode: str
) -> None:
    manifest = _write_asset_bound_identity_manifest(
        tmp_path, selection_mode=selection_mode
    )

    identity = apartment.asset_bound_bundle_acoustic_visual_identity(tmp_path)

    assert identity["status"] == "not_verified"
    assert identity["verification_status"] == "not_verified"
    assert identity["acoustic_room_ref"] is None
    assert identity["visual_room_ref"] == identity["runtime_room_ref"]
    assert identity["acoustic_selection_binding_sha256"] == manifest[
        "acoustic_selection_binding"
    ]["binding_content_sha256"]


def test_asset_bound_runtime_identity_rejects_room_or_hash_drift(
    tmp_path: Path,
) -> None:
    wrong_room_ref = {
        "registry_id": "avengine_m6_representative_rooms_v1",
        "room_id": "wrong_room",
        "revision": "v1",
    }
    _write_asset_bound_identity_manifest(
        tmp_path, visual_room_ref=wrong_room_ref
    )
    with pytest.raises(
        apartment.SpearApartmentError, match="visual_room_ref differs"
    ):
        apartment.asset_bound_bundle_acoustic_visual_identity(tmp_path)

    manifest = _write_asset_bound_identity_manifest(
        tmp_path, binding_room_ref=wrong_room_ref
    )
    with pytest.raises(
        apartment.SpearApartmentError, match="acoustic selection room_ref differs"
    ):
        apartment.asset_bound_bundle_acoustic_visual_identity(tmp_path)

    manifest["acoustic_selection_binding"]["binding_id"] = "tampered"
    (tmp_path / "manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    with pytest.raises(apartment.SpearApartmentError, match="hash is invalid"):
        apartment.asset_bound_bundle_acoustic_visual_identity(tmp_path)


def test_asset_bound_suite_carries_one_identity_and_checks_episode_sha(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _make_asset_bound_input_tree(tmp_path)
    paths["sensor_rig_trajectory"].unlink()
    manifest = _write_asset_bound_identity_manifest(tmp_path)
    binding_sha256 = manifest["acoustic_selection_binding"][
        "binding_content_sha256"
    ]
    assets = {
        "source1": CURRENT_GENERATED_DOG_ASSET_ID,
        "source2": CURRENT_GENERATED_CAT_ASSET_ID,
    }
    paths["batch_binding"].write_text(
        json.dumps(
            {
                "schema": apartment.ASSET_BOUND_EPISODE_BINDING_SCHEMA,
                "status": "pass",
                "episode_id": "episode_0001",
                "asset_ids_by_source_slot": assets,
                "acoustic_selection_binding_sha256": binding_sha256,
            }
        ),
        encoding="utf-8",
    )
    plan = _plan("episode_0001")
    for index, slot in enumerate(("source1", "source2")):
        plan["actors"][index]["actor_id"] = f"{slot}_actor"
        plan["actors"][index]["asset_id"] = assets[slot]
        for frame in plan["frames"]:
            frame["actor_states"][index]["actor_id"] = f"{slot}_actor"
    monkeypatch.setattr(
        apartment,
        "build_spear_visual_plan_from_files",
        lambda **_: deepcopy(plan),
    )

    suite = apartment.build_native_apartment_asset_bound_suite(tmp_path)

    identity = suite["acoustic_visual_identity"]
    assert identity["status"] == "pass"
    assert suite["backend_role"] == "production_visual"
    assert suite["scenarios"][0]["backend_role"] == "production_visual"
    assert suite["scenarios"][0]["plan"]["backend_role"] == "production_visual"
    assert suite["scenarios"][0]["acoustic_visual_identity"] == identity
    assert (
        suite["scenarios"][0]["native_scene"]["room_ref"]
        == identity["runtime_room_ref"]
    )

    batch = json.loads(paths["batch_binding"].read_text(encoding="utf-8"))
    batch["acoustic_selection_binding_sha256"] = "0" * 64
    paths["batch_binding"].write_text(json.dumps(batch), encoding="utf-8")
    with pytest.raises(
        apartment.SpearApartmentError,
        match="episode acoustic selection binding SHA differs",
    ):
        apartment.build_native_apartment_asset_bound_suite(tmp_path)


def test_scenario_path_discovery_is_bounded_to_s0_s3_s4(tmp_path: Path) -> None:
    expected = _make_input_tree(tmp_path, "S3")
    observed = apartment.scenario_input_paths(tmp_path, "S3")
    assert observed == {key: value.resolve() for key, value in expected.items()}
    with pytest.raises(apartment.SpearApartmentError, match="unsupported"):
        apartment.scenario_input_paths(tmp_path, "S2")


def test_motion_pilot_path_discovery_and_suite_use_p0_to_p3(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    expected = _make_motion_pilot_input_tree(tmp_path, "P0")
    observed = apartment.motion_pilot_input_paths(tmp_path, "P0")
    assert observed == {key: value.resolve() for key, value in expected.items()}
    with pytest.raises(apartment.SpearApartmentError, match="unsupported"):
        apartment.motion_pilot_input_paths(tmp_path, "S0")

    def build_plan(**kwargs: object) -> dict:
        assert kwargs["backend_role"] == apartment.BACKEND_ROLE
        return _plan("P0")

    monkeypatch.setattr(apartment, "build_spear_visual_plan_from_files", build_plan)
    suite = apartment.build_native_apartment_motion_pilot_suite(
        tmp_path, scenario_ids=("P0",)
    )
    scenario = suite["scenarios"][0]
    assert scenario["scenario_id"] == "P0"
    assert scenario["scenario_directory"] == "00_static_static"
    assert scenario["variant_id"] == "A"
    assert scenario["backend_role"] == "production_visual"
    assert suite["backend_role"] == "production_visual"
    assert suite["authority"]["spear_unreal"] == ["production RGB pixels"]


def test_asset_bound_path_discovery_carries_optional_sensor_rig_sidecar(
    tmp_path: Path,
) -> None:
    expected = _make_asset_bound_input_tree(tmp_path)

    observed = apartment.asset_bound_episode_input_paths(
        tmp_path, "episode_0001"
    )

    assert observed == {
        key: value.resolve() for key, value in expected.items()
    }
    expected["sensor_rig_trajectory"].unlink()
    legacy = apartment.asset_bound_episode_input_paths(
        tmp_path, "episode_0001"
    )
    assert "sensor_rig_trajectory" not in legacy


def _exact_binding_case(tmp_path: Path) -> tuple[dict, dict, Path]:
    asset_id = "pixel3d_cat"
    revision = "v1"
    actions = {"idle": "/Game/Test/Idle.Idle", "walk": "/Game/Test/Walk.Walk"}
    spear = {
        "actor_scale": 0.875,
        "skeletal_mesh_path": "/Game/Test/SK_Cat.SK_Cat",
        "animation_paths_by_action_id": actions,
    }
    exact = {
        "schema": apartment.EXACT_ASSET_BOUND_RUNTIME_BINDING_SCHEMA,
        "source_slot_id": "source2",
        "asset_id": asset_id,
        "asset_revision": revision,
        "actor_scale": 0.875,
        "emitter": {},
        "timeline": {
            "template_id": "cat_v1",
            "body_plan_id": "quadruped_felid_v1",
            "local_anatomical_forward_axis": [1.0, 0.0, 0.0],
            "animation_paths_by_action_id": actions,
        },
        "spear_unreal": spear,
        "asset_bound_lineage": {},
    }
    batch = {
        "schema": apartment.ASSET_BOUND_EPISODE_BINDING_SCHEMA,
        "status": "pass",
        "episode_id": "episode_0001",
        "asset_ids_by_source_slot": {
            "source1": "legacy_dog",
            "source2": asset_id,
        },
        "runtime_bindings_by_source_slot": {"source2": exact},
    }
    path = tmp_path / "batch_binding.json"
    path.write_text(json.dumps(batch), encoding="utf-8")
    plan = {
        "actors": [
            {
                "actor_id": "source2_actor",
                "asset_id": asset_id,
                "template_id": "cat_v1",
                "body_plan_id": "quadruped_felid_v1",
                "habitat_local_anatomical_forward_axis": [1.0, 0.0, 0.0],
            }
        ]
    }
    return plan, {asset_id: {**deepcopy(spear), "asset_revision": revision}}, path


def test_exact_asset_bound_snapshot_attaches_to_compiled_actor(tmp_path: Path) -> None:
    plan, bindings, batch_path = _exact_binding_case(tmp_path)

    apartment._attach_exact_asset_bound_runtime_bindings(
        plan=plan,
        scenario_id="episode_0001",
        batch_binding_path=batch_path,
        actor_bindings=bindings,
    )

    actor = plan["actors"][0]
    assert actor["actor_scale"] == pytest.approx(0.875)
    assert actor["animation_paths_by_action_id"]["walk"].endswith("Walk.Walk")
    mesh = actor["exact_runtime_binding"]["spear_unreal"]["skeletal_mesh_path"]
    assert mesh.endswith("SK_Cat.SK_Cat")


def test_legacy_asset_bound_batch_binding_remains_compatible(tmp_path: Path) -> None:
    plan, bindings, batch_path = _exact_binding_case(tmp_path)
    batch = json.loads(batch_path.read_text(encoding="utf-8"))
    batch.pop("runtime_bindings_by_source_slot")
    batch_path.write_text(json.dumps(batch), encoding="utf-8")

    apartment._attach_exact_asset_bound_runtime_bindings(
        plan=plan,
        scenario_id="episode_0001",
        batch_binding_path=batch_path,
        actor_bindings=bindings,
    )

    assert "exact_runtime_binding" not in plan["actors"][0]


def test_episode_binding_sha_must_match_bundle_identity(tmp_path: Path) -> None:
    plan, bindings, batch_path = _exact_binding_case(tmp_path)
    batch = json.loads(batch_path.read_text(encoding="utf-8"))
    batch["acoustic_selection_binding_sha256"] = "a" * 64
    batch_path.write_text(json.dumps(batch), encoding="utf-8")

    with pytest.raises(
        apartment.SpearApartmentError,
        match="episode acoustic selection binding SHA differs",
    ):
        apartment._attach_exact_asset_bound_runtime_bindings(
            plan=plan,
            scenario_id="episode_0001",
            batch_binding_path=batch_path,
            actor_bindings=bindings,
            expected_acoustic_selection_binding_sha256="b" * 64,
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda exact: exact["spear_unreal"].update(actor_scale=1.0),
            "SPEAR/UE binding differs",
        ),
        (
            lambda exact: exact["timeline"].update(template_id="wrong"),
            "compiled inputs",
        ),
    ],
)
def test_exact_asset_bound_snapshot_mismatch_fails_closed(
    tmp_path: Path, mutation, message: str
) -> None:
    plan, bindings, batch_path = _exact_binding_case(tmp_path)
    batch = json.loads(batch_path.read_text(encoding="utf-8"))
    mutation(batch["runtime_bindings_by_source_slot"]["source2"])
    batch_path.write_text(json.dumps(batch), encoding="utf-8")

    with pytest.raises(apartment.SpearApartmentError, match=message):
        apartment._attach_exact_asset_bound_runtime_bindings(
            plan=plan,
            scenario_id="episode_0001",
            batch_binding_path=batch_path,
            actor_bindings=bindings,
        )


def test_scenario_execution_keeps_native_map_and_habitat_authority(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _make_input_tree(tmp_path, "S3")
    monkeypatch.setattr(
        apartment,
        "build_spear_visual_plan_from_files",
        lambda **_: _plan("S3"),
    )
    record = apartment.build_native_apartment_scenario(tmp_path, "S3")
    assert record["native_scene"] == {
        "map": "/Game/SPEAR/Scenes/apartment_0000/Maps/apartment_0000",
        "layout": "native_map_unchanged",
        "lighting": "native_map_unchanged_no_added_lights",
        "lighting_profile": dict(apartment.NATIVE_LIGHTING_PROFILE),
        "outdoor_view": "native_map_assets_and_postprocess",
        "room_runtime_profile_id": "spear_apartment_0000",
        "room_ref": {
            "registry_id": "avengine_m6_representative_rooms_v1",
            "room_id": "legacy_ue_apartment_0000_v1",
            "revision": "real_surface_export_pending_portable_package_v1",
        },
    }
    assert record["render"] == {
        "width": 1280,
        "height": 720,
        "frame_count": 75,
        "frame_rate_hz": 15,
        "horizontal_fov_deg": 105.0,
        "streaming_warmup_frames": 120,
        "camera_warmup_frames": 40,
    }
    assert record["reuse_contract"]["audio_camera_fov_cutoff"] is False
    assert record["plan"]["authority"]["backend_may_replan"] is False
    dog = next(
        value for value in record["plan"]["actors"] if value["actor_id"] == "dog0"
    )
    assert dog["ue_component_frame_delta"] == {
        "schema": "avengine_spear_component_frame_delta_v1",
        "rotation_deg": [0.0, 90.0, 0.0],
        "translation_cm": [0.0, 0.0, 33.64],
        "composition": "add_relative_preserving_blueprint_transform",
        "reason": "exact_M2_GLTF_to_UE_asset_local_axis_and_floor_calibration",
    }
    assert record["authoritative_inputs"] == {
        key: value.relative_to(tmp_path).as_posix() for key, value in paths.items()
    }


def test_apartment_lighting_profiles_keep_native_map_and_validate_photometry() -> None:
    path = (
        Path(__file__).resolve().parents[2]
        / "examples/rooms/spear_apartment_lighting_profiles.json"
    )
    profile = apartment.load_apartment_lighting_profile(path, "warm_indoor_fill")
    assert profile["profile_id"] == "warm_indoor_fill"
    assert len(profile["generated_lights"]) == 2
    assert all(light["cast_shadows"] for light in profile["generated_lights"])

    document = {
        "schema": apartment.LIGHTING_PROFILE_SCHEMA,
        "default_profile_id": "bad",
        "profiles": [
            {
                "profile_id": "bad",
                "label": "bad",
                "claim_boundary": "test",
                "generated_lights": [
                    {
                        "light_id": "x",
                        "position_ue_cm": [0, 0, 250],
                        "intensity_lumens": -1,
                        "attenuation_radius_cm": 400,
                        "temperature_kelvin": 4000,
                    }
                ],
            }
        ],
    }
    with pytest.raises(apartment.SpearApartmentError, match="not physical"):
        apartment.resolve_apartment_lighting_profile(document)

    document["profiles"][0]["generated_lights"][0]["intensity_lumens"] = 100
    document["profiles"][0]["generated_lights"][0]["cast_shadows"] = "false"
    with pytest.raises(apartment.SpearApartmentError, match="must be boolean"):
        apartment.resolve_apartment_lighting_profile(document)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("role", "production-visual"),
        ("map", "not the native"),
        ("scenario", "disagrees"),
        ("replan", "must not replan"),
        ("hfov", "105 degree"),
        ("actors", "actor closure"),
    ],
)
def test_native_plan_validation_fails_closed(mutation: str, message: str) -> None:
    plan = _plan("S3")
    if mutation == "role":
        plan["backend_role"] = "comparison_visual"
    elif mutation == "map":
        plan["room"]["source_scene_provenance"]["scene_id"] = "proxy"
    elif mutation == "scenario":
        plan["source_logic"]["scenario_id"] = "S4"
    elif mutation == "replan":
        plan["authority"]["backend_may_replan"] = True
    elif mutation == "hfov":
        plan["camera"]["horizontal_fov_deg"] = 90.0
    elif mutation == "actors":
        plan["actors"].reverse()
    with pytest.raises(apartment.SpearApartmentError, match=message):
        apartment._validate_native_plan(plan, scenario_id="S3")


def test_animation_position_uses_normalized_timeline_phase() -> None:
    assert apartment.animation_position_seconds(0.625, 1.6) == pytest.approx(1.0)
    with pytest.raises(apartment.SpearApartmentError, match=r"\[0,1\)"):
        apartment.animation_position_seconds(1.0, 1.6)
    with pytest.raises(apartment.SpearApartmentError, match="positive"):
        apartment.animation_position_seconds(0.5, 0.0)


def test_root_readback_gate_covers_every_actor_and_camera_frame() -> None:
    plan = _plan()
    actor_readbacks = {"dog0": [], "human0": []}
    camera_readbacks = []
    for frame_index, frame in enumerate(plan["frames"]):
        for state in frame["actor_states"]:
            actor_readbacks[state["actor_id"]].append(
                {
                    "frame_index": frame_index,
                    "location_cm": list(state["translation_ue_cm"]),
                    "rotation_deg": [0.0, 0.0, state["actor_yaw_ue_deg"]],
                }
            )
        camera_readbacks.append(
            {
                "frame_index": frame_index,
                "location_cm": [-70.0, 65.0, 147.1],
                "rotation_deg": [0.0, 0.0, -145.0],
            }
        )
    summary = apartment.summarize_root_readbacks(
        expected_frames=plan["frames"],
        actor_readbacks=actor_readbacks,
        camera_readbacks=camera_readbacks,
        camera_position_cm=[-70.0, 65.0, 147.1],
        camera_yaw_deg=-145.0,
    )
    assert set(summary) == {"dog0", "human0", "camera"}
    assert all(value["status"] == "pass" for value in summary.values())

    drifted = deepcopy(actor_readbacks)
    drifted["human0"][11]["location_cm"][0] += 1.0
    with pytest.raises(apartment.SpearApartmentError, match="human0.*drifted"):
        apartment.summarize_root_readbacks(
            expected_frames=plan["frames"],
            actor_readbacks=drifted,
            camera_readbacks=camera_readbacks,
            camera_position_cm=[-70.0, 65.0, 147.1],
            camera_yaw_deg=-145.0,
        )


def test_dynamic_camera_states_drive_per_frame_readback_gate() -> None:
    plan = _dynamic_camera_plan()
    camera_states = apartment.materialize_camera_states(plan)
    assert len(camera_states) == apartment.FRAME_COUNT
    assert camera_states[0]["ue_position_cm"] != camera_states[-1][
        "ue_position_cm"
    ]

    actor_readbacks = {"dog0": [], "human0": []}
    camera_readbacks = []
    for frame_index, (frame, camera_state) in enumerate(
        zip(plan["frames"], camera_states)
    ):
        for state in frame["actor_states"]:
            actor_readbacks[state["actor_id"]].append(
                {
                    "frame_index": frame_index,
                    "location_cm": list(state["translation_ue_cm"]),
                    "rotation_deg": [
                        0.0,
                        0.0,
                        state["actor_yaw_ue_deg"],
                    ],
                }
            )
        camera_readbacks.append(
            {
                "frame_index": frame_index,
                "location_cm": list(camera_state["ue_position_cm"]),
                "rotation_deg": [0.0, 0.0, camera_state["ue_yaw_deg"]],
                "expected_pose_hash": camera_state["pose_hash"],
            }
        )

    summary = apartment.summarize_root_readbacks(
        expected_frames=plan["frames"],
        actor_readbacks=actor_readbacks,
        camera_readbacks=camera_readbacks,
    )

    assert summary["camera"]["status"] == "pass"
    assert summary["camera"]["per_frame_camera_state"] is True
    assert summary["camera"]["checked_pose_hash_count"] == 75
    assert summary["camera"]["unique_expected_pose_hash_count"] == 75

    drifted = deepcopy(camera_readbacks)
    drifted[37]["location_cm"][0] += 1.0
    with pytest.raises(
        apartment.SpearApartmentError, match="camera root readback drifted"
    ):
        apartment.summarize_root_readbacks(
            expected_frames=plan["frames"],
            actor_readbacks=actor_readbacks,
            camera_readbacks=drifted,
        )

    wrong_hash = deepcopy(camera_readbacks)
    wrong_hash[37]["expected_pose_hash"] = "0" * 64
    with pytest.raises(
        apartment.SpearApartmentError, match="pose hash differs"
    ):
        apartment.summarize_root_readbacks(
            expected_frames=plan["frames"],
            actor_readbacks=actor_readbacks,
            camera_readbacks=wrong_hash,
        )


def test_camera_state_validation_rejects_hash_and_partial_upgrade() -> None:
    plan = _dynamic_camera_plan()
    plan["frames"][12]["camera_state"]["pose_hash"] = "0" * 64
    with pytest.raises(apartment.SpearApartmentError, match="pose_hash"):
        apartment.materialize_camera_states(plan)

    partial = _dynamic_camera_plan()
    partial["frames"][12].pop("camera_state")
    with pytest.raises(apartment.SpearApartmentError, match="partially"):
        apartment.materialize_camera_states(partial)


def test_runner_applies_the_current_camera_frame_before_readback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    states = apartment.materialize_camera_states(_dynamic_camera_plan())
    applied = []
    monkeypatch.setattr(
        _RUNNER,
        "_apply_camera",
        lambda _camera, state: applied.append(state["frame_index"]),
    )
    monkeypatch.setattr(
        _RUNNER,
        "_actor_readback",
        lambda _camera, frame_index: {
            "frame_index": frame_index,
            "location_cm": [0.0, 0.0, 0.0],
            "rotation_deg": [0.0, 0.0, 0.0],
        },
    )

    first = _RUNNER._apply_camera_state_and_readback(
        object(), states[0], 0
    )
    last = _RUNNER._apply_camera_state_and_readback(
        object(), states[-1], 74
    )

    assert applied == [0, 74]
    assert first["expected_pose_hash"] == states[0]["pose_hash"]
    assert last["expected_pose_hash"] == states[-1]["pose_hash"]
    with pytest.raises(RuntimeError, match="frame order"):
        _RUNNER._apply_camera_state_and_readback(object(), states[1], 2)


def test_media_commands_copy_audio_and_reuse_only_topdown_right_panel() -> None:
    clean = apartment.build_clean_binaural_mux_command(
        ue_video_path="ue.mp4",
        authoritative_clean_path="habitat_clean.mp4",
        output_path="clean.mp4",
    )
    assert clean[clean.index("-map") + 1] == "0:v:0"
    assert "1:a:0" in clean
    assert clean[clean.index("-c:a") + 1] == "copy"
    assert "-shortest" not in clean

    topdown = apartment.build_topdown_visual_command(
        ue_video_path="ue.mp4",
        authoritative_diagnostic_path="habitat_diag.mp4",
        output_path="topdown.mp4",
    )
    graph = topdown[topdown.index("-filter_complex") + 1]
    assert "crop=640:480:iw-640:0[topdown]" in graph
    assert "[ue][topdown]hstack" in graph
    assert "-an" in topdown
    assert "-shortest" not in topdown

    nvenc = apartment.build_topdown_visual_command(
        ue_video_path="ue.mp4",
        authoritative_diagnostic_path="habitat_diag.mp4",
        output_path="topdown.mp4",
        video_encoder="h264_nvenc",
        encoder_gpu=3,
    )
    assert nvenc[nvenc.index("-c:v") + 1] == "h264_nvenc"
    assert nvenc[nvenc.index("-gpu") + 1] == "3"
    assert nvenc[nvenc.index("-preset") + 1] == "p5"

    raw = apartment.build_rawvideo_encode_command(
        output_path="raw.mp4",
        video_encoder="h264_nvenc",
        encoder_gpu=3,
    )
    assert raw[raw.index("-f") + 1] == "rawvideo"
    assert raw[raw.index("-pixel_format") + 1] == "bgr24"
    assert raw[raw.index("-video_size") + 1] == "1280x720"
    assert raw[raw.index("-framerate") + 1] == "15"
    assert raw[raw.index("-i") + 1] == "pipe:0"
    assert raw[raw.index("-frames:v") + 1] == "75"
    assert raw[raw.index("-gpu") + 1] == "3"

    with pytest.raises(apartment.SpearApartmentError, match="pixel_format"):
        apartment.build_rawvideo_encode_command(
            output_path="bad.mp4", pixel_format="rgba"
        )

    with pytest.raises(apartment.SpearApartmentError, match="unsupported"):
        apartment.build_png_encode_command(
            frames_pattern="frame_%04d.png",
            output_path="bad.mp4",
            video_encoder="unknown",
        )


def test_resume_reopens_complete_scenario_and_discards_only_incomplete_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    scenario = {"scenario_id": "episode_0001"}
    scenario_root = tmp_path / "episode_0001"
    scenario_root.mkdir()
    media = {
        media_id: {"status": "pass", "path": f"{media_id}.mp4"}
        for media_id in _RUNNER.MEDIA_EXPECTATIONS
    }
    (scenario_root / "evidence.json").write_text(
        json.dumps(
            {
                "status": "pass",
                "scenario_id": "episode_0001",
                "timing": {"video_encoder": "h264_nvenc"},
                "media": media,
            }
        ),
        encoding="utf-8",
    )

    def fake_probe(path: Path, **_kwargs: object) -> dict:
        return media[path.stem]

    monkeypatch.setattr(_RUNNER, "_probe_media", fake_probe)
    reopened = _RUNNER._load_resumable_scenario_record(
        output_root=tmp_path,
        scenario=scenario,
        video_encoder="h264_nvenc",
    )
    assert reopened is not None
    assert reopened["scenario_id"] == "episode_0001"

    incomplete = tmp_path / "episode_0002"
    incomplete.mkdir()
    (incomplete / "partial.mp4").touch()
    assert (
        _RUNNER._load_resumable_scenario_record(
            output_root=tmp_path,
            scenario={"scenario_id": "episode_0002"},
            video_encoder="h264_nvenc",
        )
        is None
    )
    assert not incomplete.exists()


def test_runner_rejects_scenario_room_or_binding_identity_drift_before_ue(
    tmp_path: Path,
) -> None:
    _write_asset_bound_identity_manifest(tmp_path)
    room_profile = deepcopy(apartment.DEFAULT_ROOM_RUNTIME_PROFILE)
    identity = apartment.asset_bound_bundle_acoustic_visual_identity(tmp_path)
    suite = {
        "room_runtime_profile": room_profile,
        "native_map": room_profile["scene"]["map_path"],
        "acoustic_visual_identity": identity,
        "scenarios": [
            {
                "scenario_id": "episode_0001",
                "native_scene": {
                    "room_runtime_profile_id": room_profile["profile_id"],
                    "room_ref": deepcopy(room_profile["room_ref"]),
                    "map": room_profile["scene"]["map_path"],
                },
                "acoustic_visual_identity": deepcopy(identity),
            }
        ],
    }

    _RUNNER._assert_suite_runtime_identity_closure(
        suite,
        input_layout="asset-bound-batch",
        room_runtime_profile=room_profile,
    )

    suite["scenarios"][0]["acoustic_visual_identity"][
        "acoustic_selection_binding_sha256"
    ] = "0" * 64
    with pytest.raises(RuntimeError, match="differs from its suite"):
        _RUNNER._assert_suite_runtime_identity_closure(
            suite,
            input_layout="asset-bound-batch",
            room_runtime_profile=room_profile,
        )


def test_resume_rejects_a_different_acoustic_visual_identity(
    tmp_path: Path,
) -> None:
    _write_asset_bound_identity_manifest(tmp_path)
    identity = apartment.asset_bound_bundle_acoustic_visual_identity(tmp_path)
    scenario_root = tmp_path / "episode_0001"
    scenario_root.mkdir()
    retained_identity = deepcopy(identity)
    retained_identity["acoustic_selection_binding_sha256"] = "0" * 64
    (scenario_root / "evidence.json").write_text(
        json.dumps(
            {
                "status": "pass",
                "scenario_id": "episode_0001",
                "timing": {"video_encoder": "h264_nvenc"},
                "acoustic_visual_identity": retained_identity,
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(
        RuntimeError, match="resumable scenario evidence is invalid"
    ):
        _RUNNER._load_resumable_scenario_record(
            output_root=tmp_path,
            scenario={
                "scenario_id": "episode_0001",
                "acoustic_visual_identity": identity,
            },
            video_encoder="h264_nvenc",
        )


def test_resume_requires_the_same_retained_execution_plan(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundle = tmp_path / "bundle"
    output = tmp_path / "output"
    bundle.mkdir()
    output.mkdir()
    retained = {"scenarios": [{"scenario_id": "old"}]}
    (output / "suite_execution_plan.json").write_text(
        json.dumps(retained), encoding="utf-8"
    )
    current = {"scenarios": [{"scenario_id": "new"}]}
    monkeypatch.setattr(
        _RUNNER, "load_apartment_lighting_profile", lambda *_args: {}
    )
    monkeypatch.setattr(
        _RUNNER, "build_native_apartment_asset_bound_suite", lambda *_args, **_kwargs: current
    )
    monkeypatch.setattr(
        _RUNNER, "_assert_suite_actor_binding_closure", lambda _suite: None
    )
    monkeypatch.setattr(
        _RUNNER,
        "_assert_suite_runtime_identity_closure",
        lambda *_args, **_kwargs: None,
    )
    args = _RUNNER.parse_args(
        [
            "--bundle-root",
            str(bundle),
            "--input-layout",
            "asset-bound-batch",
            "--spear-executable",
            str(tmp_path / "SpearSim.sh"),
            "--output-dir",
            str(output),
            "--resume",
        ]
    )
    with pytest.raises(
        RuntimeError, match="differs from the retained execution plan"
    ):
        _RUNNER.run(args)
    assert json.loads(
        (output / "suite_execution_plan.json").read_text(encoding="utf-8")
    ) == retained


def test_exact_episode_shards_are_balanced_disjoint_and_exhaustive(
    tmp_path: Path,
) -> None:
    episode_ids = tuple(f"episode_{index:04d}" for index in range(7))
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {
                "status": "pass",
                "episode_count": len(episode_ids),
                "episode_ids": list(episode_ids),
            }
        ),
        encoding="utf-8",
    )

    declared = apartment.asset_bound_bundle_episode_ids(tmp_path)
    shards = [
        apartment.contiguous_episode_shard(
            declared, shard_count=3, shard_index=index
        )
        for index in range(3)
    ]

    assert [len(shard) for shard in shards] == [3, 2, 2]
    assert tuple(value for shard in shards for value in shard) == episode_ids
    assert set(shards[0]).isdisjoint(shards[1])
    assert set(shards[0]).isdisjoint(shards[2])
    assert set(shards[1]).isdisjoint(shards[2])


def test_runner_dry_run_records_only_its_exact_manifest_shard(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundle = tmp_path / "bundle"
    output = tmp_path / "output"
    bundle.mkdir()
    episode_ids = [f"episode_{index:04d}" for index in range(5)]
    (bundle / "manifest.json").write_text(
        json.dumps(
            {
                "status": "pass",
                "episode_count": len(episode_ids),
                "episode_ids": episode_ids,
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        _RUNNER, "load_apartment_lighting_profile", lambda *_args: {}
    )

    def fake_suite(
        _bundle: Path,
        *,
        scenario_ids: tuple[str, ...],
        actor_bindings: dict,
        lighting_profile: dict,
        room_runtime_profile: dict,
    ) -> dict:
        assert lighting_profile == {}
        assert actor_bindings
        assert room_runtime_profile["profile_id"] == "spear_apartment_0000"
        return {
            "native_map": room_runtime_profile["scene"]["map_path"],
            "room_runtime_profile": room_runtime_profile,
            "scenarios": [
                {"scenario_id": scenario_id} for scenario_id in scenario_ids
            ]
        }

    monkeypatch.setattr(
        _RUNNER, "build_native_apartment_asset_bound_suite", fake_suite
    )
    monkeypatch.setattr(
        _RUNNER, "_assert_suite_actor_binding_closure", lambda _suite: None
    )
    monkeypatch.setattr(
        _RUNNER,
        "_assert_suite_runtime_identity_closure",
        lambda *_args, **_kwargs: None,
    )
    args = _RUNNER.parse_args(
        [
            "--bundle-root",
            str(bundle),
            "--input-layout",
            "asset-bound-batch",
            "--spear-executable",
            str(tmp_path / "SpearSim.sh"),
            "--output-dir",
            str(output),
            "--shard-count",
            "2",
            "--shard-index",
            "1",
            "--dry-run",
        ]
    )

    plan_path = _RUNNER.run(args)
    plan = json.loads(plan_path.read_text(encoding="utf-8"))

    assert [value["scenario_id"] for value in plan["scenarios"]] == episode_ids[3:]
    assert plan["execution_partition"] == {
        "kind": "contiguous_manifest_episode_ids",
        "shard_count": 2,
        "shard_index": 1,
        "total_episode_count": 5,
        "selected_episode_count": 2,
        "first_episode_id": "episode_0003",
        "last_episode_id": "episode_0004",
    }


@pytest.mark.parametrize(
    "argv",
    [
        ["--shard-count", "2"],
        ["--shard-index", "0"],
        ["--shard-count", "2", "--shard-index", "2"],
        ["--shard-count", "2", "--shard-index", "0", "--scenario", "x"],
    ],
)
def test_runner_rejects_incomplete_or_overlapping_shard_selection(
    tmp_path: Path, argv: list[str]
) -> None:
    with pytest.raises(SystemExit):
        _RUNNER.parse_args(
            [
                "--input-layout",
                "asset-bound-batch",
                "--spear-executable",
                str(tmp_path / "SpearSim.sh"),
                "--output-dir",
                str(tmp_path / "output"),
                *argv,
            ]
        )


@pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="FFmpeg tools are unavailable",
)
def test_media_probe_requires_full_packet_identical_binaural_copy(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.mp4"
    visual = tmp_path / "visual.mp4"
    copied = tmp_path / "copied.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "color=c=gray:s=64x48:r=15:d=5",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:sample_rate=16000:duration=5",
            "-frames:v",
            "75",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-ar",
            "16000",
            "-ac",
            "2",
            str(source),
        ],
        check=True,
    )
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "color=c=black:s=64x48:r=15:d=5",
            "-frames:v",
            "75",
            "-an",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(visual),
        ],
        check=True,
    )
    subprocess.run(
        apartment.build_clean_binaural_mux_command(
            ue_video_path=visual,
            authoritative_clean_path=source,
            output_path=copied,
        ),
        check=True,
    )

    probe = _RUNNER._probe_media(
        copied,
        expected_width=64,
        expected_height=48,
        expect_audio=True,
    )
    assert probe["size_bytes"] == copied.stat().st_size
    assert probe["audio_packet_sha256"] == _RUNNER._audio_packet_sha256(source)


def test_runtime_timing_contract_requires_rgb_and_topdown_outputs() -> None:
    assert _RUNNER.TIMING_SCHEMA == "avengine_apartment_runtime_timing_v1"
    assert _RUNNER.REQUIRED_SAMPLE_OUTPUTS == (
        "ue_visual_only.mp4",
        "ue_topdown_visual_only.mp4",
        "ue_clean_binaural.mp4",
        "ue_topdown_binaural.mp4",
    )
    started = _RUNNER.time.perf_counter()
    assert _RUNNER._elapsed_seconds(started) >= 0.0


def test_runner_closes_shared_camera_before_instance_after_render_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    events: list[object] = []
    camera = object()

    class FakeFrame:
        def __init__(self, name: str) -> None:
            self.name = name

        def __enter__(self) -> None:
            events.append((self.name, "enter"))

        def __exit__(
            self, _exc_type: object, _exc: object, _traceback: object
        ) -> bool:
            events.append((self.name, "exit"))
            return False

    class FakeCapture:
        def terminate_sp_funcs(self) -> None:
            events.append(("capture", "terminate_sp_funcs"))
            raise RuntimeError("fake cleanup failure")

        def Terminate(self) -> None:
            events.append(("capture", "Terminate"))

    class FakeUnrealService:
        @staticmethod
        def destroy_actor(*, actor: object) -> None:
            assert actor is camera
            events.append(("camera", "destroy_actor"))

    class FakeGame:
        def __init__(self) -> None:
            self.unreal_service = FakeUnrealService()

        def get_unreal_object(self, *, uclass: str) -> "FakeGame":
            assert uclass == "UGameplayStatics"
            return self

        def SetGamePaused(self, *, bPaused: bool) -> None:
            assert bPaused is False
            events.append(("game", "unpaused"))

    class FakeInstance:
        def __init__(self) -> None:
            self.game = FakeGame()

        def get_game(self) -> FakeGame:
            return self.game

        @staticmethod
        def begin_frame() -> FakeFrame:
            return FakeFrame("begin_frame")

        @staticmethod
        def end_frame() -> FakeFrame:
            return FakeFrame("end_frame")

        @staticmethod
        def step(*, num_frames: int) -> None:
            events.append(("step", num_frames))

        @staticmethod
        def close(*, force: bool) -> None:
            assert force is False
            events.append(("instance", "close"))

    instance = FakeInstance()
    capture = FakeCapture()
    room_profile = {
        "profile_id": "test_apartment",
        "supported_input_layouts": ["m6x-canary"],
        "default_lighting_profile_id": "test_lighting",
        "scene": {"map_path": "/Game/TestApartment"},
    }
    suite = {
        "native_map": "/Game/TestApartment",
        "scenarios": [
            {
                "scenario_id": "S3",
                "plan": {"actors": [], "frames": [{"actor_states": []}]},
            }
        ],
    }
    monkeypatch.setattr(
        _RUNNER, "load_source_asset_runtime_registry", lambda _path: {
            "registry_id": "test_source_assets",
            "revision": "v1",
        }
    )
    monkeypatch.setattr(_RUNNER, "spear_actor_bindings", lambda _registry: {})
    monkeypatch.setattr(
        _RUNNER,
        "load_room_runtime_profile_registry",
        lambda _path: {"registry_id": "test_rooms", "revision": "v1"},
    )
    monkeypatch.setattr(
        _RUNNER,
        "resolve_room_runtime_profile",
        lambda _registry, _profile_id: room_profile,
    )
    monkeypatch.setattr(
        _RUNNER, "load_apartment_lighting_profile", lambda *_args: {}
    )
    monkeypatch.setattr(
        _RUNNER, "build_native_apartment_suite", lambda *_args, **_kwargs: suite
    )
    monkeypatch.setattr(
        _RUNNER,
        "_assert_suite_runtime_identity_closure",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        _RUNNER, "_assert_suite_actor_binding_closure", lambda *_args: None
    )
    monkeypatch.setattr(
        _RUNNER,
        "_configure_instance",
        lambda *_args, **_kwargs: instance,
    )
    monkeypatch.setattr(
        _RUNNER, "_spawn_camera", lambda *_args: (camera, capture)
    )
    monkeypatch.setattr(
        _RUNNER, "_spawn_generated_lights", lambda *_args: []
    )
    monkeypatch.setattr(
        _RUNNER, "materialize_camera_states", lambda _plan: [{}]
    )
    monkeypatch.setattr(_RUNNER, "_apply_camera", lambda *_args: None)
    monkeypatch.setattr(
        _RUNNER, "_spawn_runtime_actors", lambda *_args: {}
    )
    monkeypatch.setattr(
        _RUNNER,
        "_destroy_runtime_actors",
        lambda *_args: events.append(("actors", "destroy")),
    )

    def fail_render(**_kwargs: object) -> None:
        events.append(("render", "error"))
        raise RuntimeError("fake render failure")

    monkeypatch.setattr(_RUNNER, "_render_scenario", fail_render)
    args = _RUNNER.parse_args(
        [
            "--bundle-root",
            str(tmp_path / "bundle"),
            "--input-layout",
            "m6x-canary",
            "--scenario",
            "S3",
            "--spear-executable",
            str(tmp_path / "SpearSim.sh"),
            "--output-dir",
            str(tmp_path / "output"),
        ]
    )

    with pytest.raises(RuntimeError, match="fake render failure"):
        _RUNNER.run(args)

    render = events.index(("render", "error"))
    terminate_sp_funcs = events.index(("capture", "terminate_sp_funcs"))
    terminate = events.index(("capture", "Terminate"))
    destroy_camera = events.index(("camera", "destroy_actor"))
    close = events.index(("instance", "close"))
    assert render < terminate_sp_funcs < terminate < destroy_camera < close
    assert events[terminate_sp_funcs - 3 : close + 1] == [
        ("begin_frame", "enter"),
        ("begin_frame", "exit"),
        ("end_frame", "enter"),
        ("capture", "terminate_sp_funcs"),
        ("capture", "Terminate"),
        ("camera", "destroy_actor"),
        ("end_frame", "exit"),
        ("instance", "close"),
    ]


def test_scene_capture_warmup_discards_streaming_transition(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    values = [24] * 35 + [32, 48, 72, 96, 120] + [120] * 12

    class FakeInstance:
        @staticmethod
        def begin_frame():
            return nullcontext()

        @staticmethod
        def end_frame():
            return nullcontext()

    class FakeCapture:
        def __init__(self) -> None:
            self.index = 0

        def read_pixels(self) -> dict:
            value = values[min(self.index, len(values) - 1)]
            self.index += 1
            return {
                "arrays": {
                    "data": np.full(
                        (apartment.HEIGHT, apartment.WIDTH, 3),
                        value,
                        dtype=np.uint8,
                    )
                }
            }

    monkeypatch.setattr(_RUNNER, "_apply_actor_state", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(_RUNNER, "_apply_camera", lambda *_args, **_kwargs: None)
    result = _RUNNER._capture_warmup_until_stable(
        instance=FakeInstance(),
        camera=object(),
        capture=FakeCapture(),
        runtimes={"source1_actor": {}},
        actor_states=({"actor_id": "source1_actor"},),
        camera_plan={},
        minimum_frames=40,
        maximum_frames=60,
        stable_transitions=4,
        mean_abs_change_threshold=0.1,
    )

    assert result["status"] == "pass"
    assert result["mode"] == "discarded_scene_capture_readbacks"
    assert result["discarded_frame_count"] == 44
    assert result["maximum_mean_abs_change"] == pytest.approx(24.0)
    assert result["first_to_last_mean_abs_change"] == pytest.approx(96.0)


def test_default_asset_forward_bindings_are_explicit() -> None:
    assert (
        apartment.DEFAULT_ACTOR_BINDINGS[apartment.BEAGLE_ASSET_ID][
            "ue_anatomical_forward_yaw_deg"
        ]
        == 180.0
    )
    assert (
        apartment.DEFAULT_ACTOR_BINDINGS[apartment.HUMAN_ASSET_ID][
            "ue_anatomical_forward_yaw_deg"
        ]
        == 90.0
    )
    assert (
        "Standing_Idle"
        in apartment.DEFAULT_ACTOR_BINDINGS[apartment.HUMAN_ASSET_ID]["idle_animation"]
    )
    assert apartment.DEFAULT_ACTOR_BINDINGS[apartment.BEAGLE_ASSET_ID][
        "ue_component_frame_delta"
    ]["rotation_deg"] == [0.0, 90.0, 0.0]
    assert apartment.DEFAULT_ACTOR_BINDINGS[apartment.BEAGLE_ASSET_ID][
        "ue_component_frame_delta"
    ]["translation_cm"] == [0.0, 0.0, 33.64]
    assert apartment.DEFAULT_ACTOR_BINDINGS[apartment.HUMAN_ASSET_ID][
        "ue_component_frame_delta"
    ]["translation_cm"] == [0.0, 0.0, 0.0]
    current_dog = apartment.DEFAULT_ACTOR_BINDINGS[
        CURRENT_GENERATED_DOG_ASSET_ID
    ]
    assert current_dog["ue_anatomical_forward_yaw_deg"] == 0.0
    assert current_dog["ue_component_frame_delta"]["rotation_deg"] == [
        0.0,
        0.0,
        0.0,
    ]
    assert current_dog["ue_component_frame_delta"]["translation_cm"] == [
        0.0,
        0.0,
        0.0,
    ]
    assert current_dog["ue_anatomical_basis_bones"] == {
        "rear": "bone_0",
        "front": "bone_5",
        "body": "bone_0",
        "left_foot": "bone_19",
        "right_foot": "bone_23",
    }
    assert apartment.anatomical_basis_bones_for_asset(
        CURRENT_GENERATED_DOG_ASSET_ID
    ) == current_dog["ue_anatomical_basis_bones"]
    cat = apartment.DEFAULT_ACTOR_BINDINGS[CURRENT_GENERATED_CAT_ASSET_ID]
    assert cat["ue_anatomical_forward_yaw_deg"] == 0.0
    assert cat["ue_anatomical_basis_bones"] == {
        "rear": "bone_0",
        "front": "bone_3",
        "body": "bone_0",
        "left_foot": "bone_21",
        "right_foot": "bone_26",
    }
    assert cat["ue_component_frame_delta"]["translation_cm"] == [
        0.0,
        0.0,
        0.0,
    ]
    assert apartment.anatomical_basis_bones_for_asset(
        CURRENT_GENERATED_CAT_ASSET_ID
    ) == cat["ue_anatomical_basis_bones"]
    assert (
        apartment.anatomical_basis_bones_for_asset(apartment.HUMAN_ASSET_ID)
        is None
    )


def test_generated_anatomical_basis_mapping_is_exact_and_asset_local() -> None:
    bindings = deepcopy(apartment.DEFAULT_ACTOR_BINDINGS)
    bindings[CURRENT_GENERATED_DOG_ASSET_ID]["ue_anatomical_basis_bones"].pop(
        "front"
    )
    with pytest.raises(apartment.SpearApartmentError, match="define exactly"):
        apartment.anatomical_basis_bones_for_asset(
            CURRENT_GENERATED_DOG_ASSET_ID, actor_bindings=bindings
        )


def test_generated_anatomical_basis_mapping_reaches_runtime_plan(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _make_input_tree(tmp_path, "S3")
    plan = _plan("S3")
    plan["actors"][0]["asset_id"] = CURRENT_GENERATED_DOG_ASSET_ID
    monkeypatch.setattr(
        apartment,
        "build_spear_visual_plan_from_files",
        lambda **_: deepcopy(plan),
    )
    record = apartment.build_native_apartment_scenario(tmp_path, "S3")
    dog = next(
        value
        for value in record["plan"]["actors"]
        if value["actor_id"] == "dog0"
    )
    assert dog["ue_anatomical_basis_bones"]["front"] == "bone_5"


def test_component_frame_delta_must_preserve_blueprint_transform(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _make_input_tree(tmp_path, "S3")
    monkeypatch.setattr(
        apartment,
        "build_spear_visual_plan_from_files",
        lambda **_: _plan("S3"),
    )
    bindings = deepcopy(apartment.DEFAULT_ACTOR_BINDINGS)
    bindings[apartment.BEAGLE_ASSET_ID]["ue_component_frame_delta"]["composition"] = (
        "replace_blueprint_transform"
    )
    with pytest.raises(apartment.SpearApartmentError, match="may not replace"):
        apartment.build_native_apartment_scenario(
            tmp_path, "S3", actor_bindings=bindings
        )


def test_runtime_component_delta_is_added_to_authored_blueprint_transform() -> None:
    class FakeComponent:
        def __init__(self) -> None:
            self.location = {"X": 1.0, "Y": -2.0, "Z": 3.0}
            self.rotation = {"Roll": 4.0, "Pitch": 5.0, "Yaw": 6.0}

        def get_property_value(self, *, property_name: str):
            if property_name == "RelativeLocation":
                return dict(self.location)
            if property_name == "RelativeRotation":
                return dict(self.rotation)
            raise AssertionError(property_name)

        def K2_AddRelativeLocation(self, *, DeltaLocation, **_):
            for axis in ("X", "Y", "Z"):
                self.location[axis] += DeltaLocation[axis]

        def K2_AddRelativeRotation(self, *, DeltaRotation, **_):
            for axis in ("Roll", "Pitch", "Yaw"):
                self.rotation[axis] += DeltaRotation[axis]

    declaration = {
        "actor_id": "dog0",
        "asset_id": apartment.BEAGLE_ASSET_ID,
        "ue_component_frame_delta": apartment.DEFAULT_ACTOR_BINDINGS[
            apartment.BEAGLE_ASSET_ID
        ]["ue_component_frame_delta"],
    }
    result = apartment.apply_ue_component_frame_delta(FakeComponent(), declaration)
    assert result["blueprint_relative_before"] == {
        "translation_cm": [1.0, -2.0, 3.0],
        "rotation_deg": [4.0, 5.0, 6.0],
    }
    assert result["blueprint_relative_after"] == {
        "translation_cm": [1.0, -2.0, 36.64],
        "rotation_deg": [4.0, 95.0, 6.0],
    }
    assert result["timeline_anchor_mutated"] is False
    assert result["target"] == "attached_visual_actor_root_component"


def test_runtime_component_delta_accepts_equivalent_gimbal_rotator_readback() -> None:
    class GimbalComponent:
        def __init__(self) -> None:
            self.location = {"X": 0.0, "Y": 0.0, "Z": 0.0}
            self.rotation = {"Roll": 0.0, "Pitch": 0.0, "Yaw": 0.0}

        def get_property_value(self, *, property_name: str):
            if property_name == "RelativeLocation":
                return dict(self.location)
            if property_name == "RelativeRotation":
                return dict(self.rotation)
            raise AssertionError(property_name)

        def K2_AddRelativeLocation(self, *, DeltaLocation, **_):
            for axis in ("X", "Y", "Z"):
                self.location[axis] += DeltaLocation[axis]

        def K2_AddRelativeRotation(self, *, DeltaRotation, **_):
            assert DeltaRotation == {"Roll": 0.0, "Pitch": 90.0, "Yaw": 0.0}
            # UE may report the same +90 degree pitch as this Euler triplet.
            self.rotation = {"Roll": 180.0, "Pitch": 90.0, "Yaw": 180.0}

    declaration = {
        "actor_id": "dog0",
        "asset_id": apartment.BEAGLE_ASSET_ID,
        "ue_component_frame_delta": apartment.DEFAULT_ACTOR_BINDINGS[
            apartment.BEAGLE_ASSET_ID
        ]["ue_component_frame_delta"],
    }
    result = apartment.apply_ue_component_frame_delta(GimbalComponent(), declaration)
    assert result["euler_component_rotation_delta_error_deg"] == pytest.approx(180.0)
    assert result["quaternion_equivalence_rotation_error_deg"] == pytest.approx(0.0)
    assert result["maximum_rotation_delta_error_deg"] == pytest.approx(0.0)


def test_visual_bounds_gate_proves_beagle_floor_contact_and_horizontal_frame() -> None:
    plan = _plan()
    records = {"dog0": [], "human0": []}
    for frame_index, frame in enumerate(plan["frames"]):
        dog_root = frame["actor_states"][0]["translation_ue_cm"]
        human_root = frame["actor_states"][1]["translation_ue_cm"]
        records["dog0"].append(
            {
                "frame_index": frame_index,
                "minimum_cm": [dog_root[0] - 35.0, dog_root[1] - 25.0, dog_root[2]],
                "maximum_cm": [
                    dog_root[0] + 35.0,
                    dog_root[1] + 25.0,
                    dog_root[2] + 50.0,
                ],
            }
        )
        records["human0"].append(
            {
                "frame_index": frame_index,
                "minimum_cm": [
                    human_root[0] - 20.0,
                    human_root[1] - 20.0,
                    human_root[2],
                ],
                "maximum_cm": [
                    human_root[0] + 20.0,
                    human_root[1] + 20.0,
                    human_root[2] + 175.0,
                ],
            }
        )
    summary = apartment.summarize_actor_bounds(
        expected_frames=plan["frames"],
        actor_declarations=plan["actors"],
        actor_bounds=records,
    )
    assert summary["dog0"]["status"] == "pass"
    assert summary["dog0"]["maximum_floor_error_cm"] == 0.0
    assert summary["human0"]["status"] == "observed"

    current_dog_plan = deepcopy(plan)
    current_dog_plan["actors"][0]["asset_id"] = CURRENT_GENERATED_DOG_ASSET_ID
    current_dog_summary = apartment.summarize_actor_bounds(
        expected_frames=current_dog_plan["frames"],
        actor_declarations=current_dog_plan["actors"],
        actor_bounds=records,
    )
    assert current_dog_summary["dog0"]["status"] == "pass"

    drifted = deepcopy(records)
    drifted["dog0"][4]["minimum_cm"][2] -= 6.0
    with pytest.raises(apartment.SpearApartmentError, match="actor-root floor"):
        apartment.summarize_actor_bounds(
            expected_frames=plan["frames"],
            actor_declarations=plan["actors"],
            actor_bounds=drifted,
        )


def test_anatomical_forward_gate_rejects_a_visually_reversed_skeleton() -> None:
    plan = _plan()
    readbacks = {
        "dog0": [
            {
                "frame_index": frame_index,
                "basis_kind": "prefixed_bip_quadruped_longitudinal_v1",
                "forward_vector_ue": [0.0, -1.0, 0.0],
                "bone_names": {"rear": "beagle Pelvis", "front": "beagle Spine2"},
            }
            for frame_index in (0, 37, 74)
        ],
        "human0": [
            {
                "frame_index": frame_index,
                "basis_kind": "humanoid_semantic_v1",
                "forward_vector_ue": [0.573576436, -0.819152044, 0.0],
                "bone_names": {"pelvis": "Bip01 Pelvis", "spine": "Bip01 Spine2"},
            }
            for frame_index in (0, 37, 74)
        ],
    }
    summary = apartment.summarize_anatomical_forward_readbacks(
        expected_frames=plan["frames"],
        visual_forward_readbacks=readbacks,
    )
    assert summary["dog0"]["status"] == "pass"
    assert summary["dog0"]["maximum_angular_error_deg"] == pytest.approx(0.0)

    reversed_readbacks = deepcopy(readbacks)
    reversed_readbacks["dog0"][1]["forward_vector_ue"] = [0.0, 1.0, 0.0]
    with pytest.raises(apartment.SpearApartmentError, match="faces away"):
        apartment.summarize_anatomical_forward_readbacks(
            expected_frames=plan["frames"],
            visual_forward_readbacks=reversed_readbacks,
        )

    tilted_readbacks = deepcopy(readbacks)
    tilted_readbacks["dog0"][1]["forward_vector_ue"] = [
        0.0,
        -0.1,
        0.994987437,
    ]
    with pytest.raises(apartment.SpearApartmentError, match="not horizontal"):
        apartment.summarize_anatomical_forward_readbacks(
            expected_frames=plan["frames"],
            visual_forward_readbacks=tilted_readbacks,
        )

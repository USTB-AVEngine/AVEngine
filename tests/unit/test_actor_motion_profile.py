from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from avengine.qa.actor_motion_profile import (
    ActorMotionProfileError,
    bind_planning_episode,
    build_actor_motion_profile,
    build_actor_motion_profile_from_planning,
    materialize_profile_frames,
    source_center_paths,
    validate_actor_motion_authorities,
    validate_actor_motion_profile,
)

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(params=["target", "distractor", "both"])
def real_authorities(request: pytest.FixtureRequest):
    paths = {
        "target": (
            ROOT
            / "examples/qa/native_strict_two_human_target_moves_native_rate_candidate_v1.json",
            ROOT
            / "tmp/lead_a_strict_two_human_full_episode_batch_v1/dynamic_target_moves_v2_cpu_candidate_v1/target_moves_v2_preflight.json",
            ROOT
            / "tmp/lead_a_strict_two_human_full_episode_batch_v1/dynamic_target_moves_v2_materialized_v1/suite_execution_plan.json",
        ),
        "distractor": (
            ROOT
            / "examples/qa/native_strict_two_human_distractor_moves_native_rate_candidate_v1.json",
            ROOT
            / "tmp/lead_a_strict_two_human_full_episode_batch_v1/dynamic_distractor_moves_v2_geometry_v1/distractor_moves_v2_preflight.json",
            ROOT
            / "tmp/lead_a_strict_two_human_full_episode_batch_v1/dynamic_distractor_moves_v2_materialized_v1/suite_execution_plan.json",
        ),
        "both": (
            ROOT
            / "examples/qa/native_strict_two_human_both_move_native_rate_candidate_v1.json",
            ROOT
            / "tmp/lead_a_strict_two_human_full_episode_batch_v1/dynamic_both_move_v1_adapter_v1/preflight.json",
            ROOT
            / "tmp/lead_a_strict_two_human_full_episode_batch_v1/dynamic_both_move_v1_materialized_v1/suite_execution_plan.json",
        ),
    }
    candidate_path, old_path, suite_path = paths[request.param]
    if not all(path.is_file() for path in (candidate_path, old_path, suite_path)):
        pytest.skip(
            "legacy actor-motion authorities are workspace artifacts not shipped "
            "in a fresh checkout"
        )
    candidate = json.loads(candidate_path.read_text())
    old = json.loads(old_path.read_text())
    suite = json.loads(suite_path.read_text())
    return candidate_path, candidate, old_path, old["canaries"][0], suite_path, suite


def _profile(authorities):
    candidate_path, candidate, old_path, old_row, suite_path, suite = authorities
    return build_actor_motion_profile(
        candidate_path=candidate_path,
        candidate=candidate,
        old_preflight_path=old_path,
        selected_old_row=old_row,
        base_suite_path=suite_path,
        base_suite=suite,
    )


def test_planning_episode_binding_uses_absolute_file_and_unique_selector(tmp_path):
    row = {"episode_id": "episode_001", "mechanism": "target_moves"}
    path = tmp_path / "planning.json"
    path.write_text(json.dumps({"episodes": [row]}))

    binding = bind_planning_episode(
        planning_manifest_path=path,
        episode_id="episode_001",
    )

    assert binding == {
        "path": str(path.resolve()),
        "json_pointer": "/episodes/0",
        "value": row,
    }


def test_planning_episode_binding_rejects_missing_or_duplicate_selector(tmp_path):
    row = {"episode_id": "episode_001", "mechanism": "target_moves"}
    path = tmp_path / "planning.json"
    path.write_text(json.dumps({"episodes": [row]}))
    with pytest.raises(ActorMotionProfileError, match="selector"):
        bind_planning_episode(
            planning_manifest_path=path,
            episode_id="episode_999",
        )

    path.write_text(json.dumps({"episodes": [row, row]}))
    with pytest.raises(ActorMotionProfileError, match="selector"):
        bind_planning_episode(
            planning_manifest_path=path,
            episode_id="episode_001",
        )


def test_three_real_profiles_bind_frames_sources_and_stride_one_rirs(real_authorities):
    profile = _profile(real_authorities)
    validate_actor_motion_profile(profile)
    assert profile["frames"] == materialize_profile_frames(profile)
    assert source_center_paths(profile)
    assert profile["rir_expectation"]["stride_frames"] == 1


def test_profile_hash_rejects_mutation(real_authorities):
    profile = _profile(real_authorities)
    forged = copy.deepcopy(profile)
    forged["authorities"]["candidate"]["value"]["mechanism"] = "forged"
    with pytest.raises(ActorMotionProfileError, match="content hash"):
        validate_actor_motion_profile(forged)


def test_three_real_authorities_pass_semantic_closure(real_authorities):
    _, candidate, _, old_row, _, suite = real_authorities
    validate_actor_motion_authorities(candidate, old_row, suite)


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        ("active_boundary", "outside roots"),
        ("declaration", "actor declaration"),
        ("old_asset", "asset/revision"),
        ("audio", "audio role/timing"),
        ("camera", "camera cross-authority"),
        ("source_activation", "source activation"),
    ],
)
def test_semantic_validator_rejects_authority_drift(real_authorities, mutation, match):
    _, candidate_value, _, old_row_value, _, suite_value = real_authorities
    candidate = copy.deepcopy(candidate_value)
    old_row = copy.deepcopy(old_row_value)
    suite = copy.deepcopy(suite_value)
    actors = candidate["actors"]
    moving_slot = next(slot for slot, actor in actors.items() if actor["moving"])

    if mutation == "active_boundary":
        actor = actors[moving_slot]
        start = actor["native_rate_active_interval"]["output_frame_range_inclusive"][0]
        actor["root_path_m"][start - 1][0] += 0.01
        actor["translation_ue_cm_path"][start - 1][0] += 1.0
        state = candidate["frames"][start - 1]["actor_states"][
            list(actors).index(moving_slot)
        ]
        state["translation_m"][0] += 0.01
        state["translation_ue_cm"][0] += 1.0
    elif mutation == "declaration":
        actor_id = actors[moving_slot]["actor_id"]
        candidate["actor_declarations"][actor_id]["asset_id"] = "forged"
    elif mutation == "old_asset":
        role = "target" if candidate["target_slot"] == moving_slot else "distractor"
        old_row[role]["runtime_revision"] = "forged"
    elif mutation == "audio":
        candidate["audio_event_contract"]["speech_frame_window_inclusive"][0] += 1
    elif mutation == "camera":
        suite["scenarios"][0]["plan"]["camera"]["horizontal_fov_deg"] += 1
    elif mutation == "source_activation":
        candidate["source_activation_contract"]["source_logic"]["sources"][0][
            "activation"
        ] = "silent"

    with pytest.raises(ActorMotionProfileError, match=match):
        validate_actor_motion_authorities(candidate, old_row, suite)


def test_static_semantics_reject_root_motion(real_authorities):
    _, candidate_value, _, old_row, _, suite = real_authorities
    candidate = copy.deepcopy(candidate_value)
    static_slots = [
        slot for slot, actor in candidate["actors"].items() if not actor["moving"]
    ]
    if not static_slots:
        pytest.skip("real authority has no static actor")
    slot = static_slots[0]
    candidate["actors"][slot]["root_path_m"][-1][0] += 0.01
    candidate["frames"][-1]["actor_states"][list(candidate["actors"]).index(slot)][
        "translation_m"
    ][0] += 0.01
    with pytest.raises(ActorMotionProfileError, match="static actor"):
        validate_actor_motion_authorities(candidate, old_row, suite)


def _write_planning_fixture(tmp_path: Path) -> tuple[Path, Path, dict]:
    female_id = "lead_b_rocketbox_adults_female_adult_01_original_v1"
    male_id = "rocketbox_human_male_adult_01_m5_1_candidate"
    scenario_id = "fixture_native_source"
    female_roots = [[0.0, 0.4, 3.0 - 0.04 * index] for index in range(75)]
    male_root = [2.0, 0.4, 3.0]
    suite = {
        "native_map": "/Game/Apartment/Maps/apartment_0000",
        "scenarios": [
            {
                "scenario_id": scenario_id,
                "native_scene": {
                    "map": "/Game/Apartment/Maps/apartment_0000",
                    "layout": "apartment_0000",
                    "lighting": "day",
                    "lighting_profile": {"profile_id": "apartment_day"},
                    "outdoor_view": "city",
                },
                "render": {"width": 640, "height": 480},
                "plan": {
                    "room": {"room_id": "apartment_0000"},
                    "coordinate_contract": {"world": "habitat"},
                    "frames": [
                        {
                            "actor_states": [
                                {
                                    "actor_id": "native_static",
                                    "action_id": "idle",
                                    "translation_m": male_root,
                                    "anatomical_forward_habitat_world": [0, 0, -1],
                                },
                                {
                                    "actor_id": "native_moving",
                                    "action_id": "walk",
                                    "translation_m": root,
                                    "anatomical_forward_habitat_world": [0, 0, -1],
                                },
                            ]
                        }
                        for root in female_roots
                    ],
                },
            }
        ],
    }
    suite_path = tmp_path / "source_suite.json"
    suite_path.write_text(json.dumps(suite))

    def runtime_record(asset_id: str, revision: str, period: int, height: float):
        is_female = asset_id == female_id
        manifest_schema = (
            "rocketbox_batch_native_ue_import_v1"
            if is_female
            else "rocketbox_native_ue_import_v3"
        )
        tag = (
            "rocketbox_adults_female_adult_01_original_ue_v1"
            if is_female
            else "rocketbox_male_adult_01_original_ue_v3"
        )
        import_asset_id = (
            "rocketbox_female_adult_01" if is_female else "rocketbox_male_adult_01"
        )
        runtime_root = (
            "rocketbox_batch_native_runtime_ue_v1"
            if is_female
            else "rocketbox_native_runtime_ue_v3"
        )
        manifest_path = tmp_path / manifest_schema / tag / "ue_import_manifest.json"
        source_glb = tmp_path / runtime_root / tag / "runtime.glb"
        source_glb.parent.mkdir(parents=True, exist_ok=True)
        source_glb.write_bytes(b"fixture runtime")
        mesh_directory = f"/Game/{asset_id}"
        blueprint = f"/Game/Blueprints/BP_{tag}"
        idle_animation = f"{mesh_directory}/Standing_Idle.Standing_Idle"
        walking_animation = f"{mesh_directory}/Walking.Walking"
        manifest = {
            "schema": manifest_schema,
            "tag": tag,
            "asset_id": import_asset_id,
            "usage_scope": "research_candidate",
            "formal_registration_authorized": False,
            "source_glb": str(source_glb),
            "reload_verification": {"status": "passed"},
            "runtime_contract": {
                "actor_scale": 1.0,
                "bone_count": 80,
                "bounds": {"height_passed": True, "ground_passed": True},
            },
            "glb_contract": {
                "armature_scale": [1.0, 1.0, 1.0],
                "armature_translation": [0.0, 0.0, 0.0],
                "animation_names": ["Standing_Idle", "Walking"],
                "joint_count": 80,
                "skin_count": 1,
                "mesh_count": 1,
                "mesh_is_scene_root": True,
            },
            "content": {
                "blueprint": blueprint,
                "skeletal_mesh": f"{mesh_directory}/runtime.runtime",
                "skeleton": f"{mesh_directory}/runtime_Skeleton.runtime_Skeleton",
                "animations": {
                    "Standing_Idle": idle_animation,
                    "Walking": walking_animation,
                },
            },
        }
        ref = {
            "path": str(manifest_path),
            "schema": manifest_schema,
            "tag": tag,
            "import_asset_id": import_asset_id,
        }
        if is_female:
            manifest["base_avatar_id"] = "rocketbox_adults_female_adult_01"
            ref["base_avatar_id"] = "rocketbox_adults_female_adult_01"
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(manifest))
        return {
            "asset_id": asset_id,
            "revision": revision,
            "display_label": asset_id,
            "entity_class": "articulated_human",
            "identity": {"species_id": "human"},
            "realized_attributes": {"life_stage": "adult"},
            "geometry": {"source_mesh_uri": f"artifact://{asset_id}/runtime.glb"},
            "timeline": {
                "template_id": asset_id,
                "body_plan_id": "biped_human",
                "local_anatomical_forward_axis": [0.0, 0.0, 1.0],
                "walk_phase_period_frames": period,
                "idle_action_id": "idle",
                "walking_action_id": "walk",
            },
            "default_emitter_anchor_id": "mouth",
            "emitter_anchors": [
                {
                    "anchor_id": "mouth",
                    "offset_m": [0.0, height, 0.0],
                    "offset_space": "final_scaled_asset_root",
                }
            ],
            "runtime_backends": {
                "spear_unreal": {
                    "actor_scale": 1.0,
                    "ue_import_manifest_ref": ref,
                    "blueprint_class_path": f"{blueprint}.BP_{tag}_C",
                    "skeletal_mesh_binding": "blueprint_component",
                    "skeletal_mesh_path": None,
                    "idle_animation": idle_animation,
                    "walking_animation": walking_animation,
                    "ue_anatomical_forward_yaw_deg": 90.0,
                    "ue_component_frame_delta": {
                        "schema": "avengine_spear_component_frame_delta_v1",
                        "rotation_deg": [0.0, 0.0, 0.0],
                        "translation_cm": [0.0, 0.0, 0.0],
                        "composition": "add_relative_preserving_blueprint_transform",
                        "reason": "fixture identity delta",
                    },
                    "floor_contact_gate": False,
                }
            },
            "admission_state": "research",
        }

    registry = {
        "schema": "avengine_source_asset_runtime_registry_v1",
        "registry_id": "fixture_registry",
        "revision": "fixture_v1",
        "assets": [
            runtime_record(female_id, "native_runtime_ue_v1", 19, 1.57),
            runtime_record(male_id, "native_runtime_ue_v3", 16, 1.61),
        ],
    }
    registry_path = tmp_path / "runtime_registry.json"
    registry_path.write_text(json.dumps(registry))

    def projection(record: dict) -> dict:
        timeline = record["timeline"]
        spear = record["runtime_backends"]["spear_unreal"]
        anchor = record["emitter_anchors"][0]
        return {
            "schema": "avengine_global100_runtime_motion_declaration_v1",
            "runtime_registry": str(registry_path.resolve()),
            "registry_id": registry["registry_id"],
            "registry_revision": registry["revision"],
            "asset_id": record["asset_id"],
            "asset_revision": record["revision"],
            "body_plan_id": timeline["body_plan_id"],
            "template_id": timeline["template_id"],
            "local_anatomical_forward_axis": timeline["local_anatomical_forward_axis"],
            "idle_action_id": "idle",
            "walking_action_id": "walk",
            "walk_phase_period_frames": timeline["walk_phase_period_frames"],
            "actor_scale": 1.0,
            "ue_import_manifest_ref": spear["ue_import_manifest_ref"],
            "animation_paths_by_action_id": {
                "idle": spear["idle_animation"],
                "walk": spear["walking_animation"],
            },
            "blueprint_class_path": spear["blueprint_class_path"],
            "skeletal_mesh_binding": spear["skeletal_mesh_binding"],
            "skeletal_mesh_path": spear["skeletal_mesh_path"],
            "ue_anatomical_forward_yaw_deg": 90.0,
            "ue_component_frame_delta": spear["ue_component_frame_delta"],
            "floor_contact_gate": False,
            "source_mesh_uri": record["geometry"]["source_mesh_uri"],
            "emitter_anchor_id": "mouth",
            "emitter_offset_m": anchor["offset_m"],
            "emitter_offset_space": anchor["offset_space"],
            "admission_state": "research",
        }

    records = {record["asset_id"]: record for record in registry["assets"]}

    def authority(asset_id: str, actor: str, frame_map: list[int]) -> dict:
        return {
            "schema": "avengine_global100_role_motion_profile_authority_v1",
            "source_path": {
                "source_suite": str(suite_path.resolve()),
                "native_source_scenario_id": scenario_id,
                "source_actor_id": actor,
                "frame_index_map": frame_map,
                "root_path_policy": "planning_row_exact_selected_native_roots_v1",
            },
            "runtime": projection(records[asset_id]),
        }

    row = {
        "episode_id": "strict2h_full75_fixture_v1",
        "mechanism": "target_moves",
        "formal": False,
        "qualification_claim": False,
        "timeline": {"frame_count": 75, "frame_rate_hz": 15},
        "camera": {
            "translation_m": [1.0, 1.47, 0.0],
            "yaw_path_deg": [0.0] * 75,
        },
        "target": {
            "source_slot_id": "source1",
            "source_actor_id": "native_moving",
            "runtime_asset_id": female_id,
            "runtime_revision": "native_runtime_ue_v1",
            "side": "left",
            "root_path_m": female_roots,
            "frame_index_map": list(range(75)),
            "speech_frame_window_inclusive": [7, 50],
            "sound_asset_id": "fixture_speech",
            "voice_id": "fixture_voice",
            "content_id": "fixture_statement",
            "speech_sample_count": 45_912,
            "speech_sample_rate_hz": 16_000,
            "speech_channel_count": 1,
            "speech_audio_uri": "semantic://fixture_speech",
            "motion_profile_authority": authority(
                female_id, "native_moving", list(range(75))
            ),
        },
        "distractor": {
            "source_slot_id": "source2",
            "source_actor_id": "native_static",
            "runtime_asset_id": male_id,
            "runtime_revision": "native_runtime_ue_v3",
            "root_path_m": [male_root] * 75,
            "frame_index_map": [0] * 75,
            "motion_profile_authority": authority(male_id, "native_static", [0] * 75),
        },
        "audio_program": {
            "mode": "one_active_of_n",
            "active_source_slots": ["source1"],
            "silent_source_slots": ["source2"],
            "target_event": {
                "sound_asset_id": "fixture_speech",
                "voice_id": "fixture_voice",
                "content_id": "fixture_statement",
                "start_sample": 7467,
                "end_sample_exclusive": 53_379,
                "source_sample_rate_hz": 16_000,
                "source_channel_count": 1,
                "source_sample_count": 45_912,
                "source_audio_uri": "semantic://fixture_speech",
            },
        },
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps({"episodes": [row]}))
    return manifest_path, registry_path, row


def test_planning_row_profile_uses_role_specific_source_and_female_period(tmp_path):
    manifest_path, _, row = _write_planning_fixture(tmp_path)
    profile = build_actor_motion_profile_from_planning(
        planning_manifest_path=manifest_path,
        episode_id=row["episode_id"],
    )
    validate_actor_motion_profile(profile)
    actors = profile["authorities"]["candidate"]["value"]["actors"]

    assert actors["source1"]["actor_id"] == "source1_actor"
    assert (
        actors["source1"]["planning_source_path_authority"]["source_actor_id"]
        == "native_moving"
    )
    assert actors["source1"]["action_id_path"] == ["walk"] * 75
    assert actors["source1"]["action_phase_path"][:21] == [
        (index % 19) / 19 for index in range(21)
    ]
    assert actors["source1"]["action_time_ticks_path"] == [
        index * 3200 for index in range(75)
    ]
    assert actors["source1"]["root_path_m"] == row["target"]["root_path_m"]
    assert (
        actors["source1"]["anatomical_forward_habitat_world_path"]
        == [[0.0, 0.0, -1.0]] * 75
    )
    assert actors["source1"]["actor_yaw_ue_deg_path"] == pytest.approx([-180.0] * 75)
    assert actors["source1"]["trajectory_preflight"]["walk_phase_period_frames"] == 19
    claim_boundary = actors["source1"]["trajectory_preflight"]["claim_boundary"]
    assert "human motion speed" in claim_boundary
    assert "human stride" in claim_boundary
    assert "foot contact" in claim_boundary
    assert "visual acceptance" in claim_boundary
    assert "pending native capture and review" in claim_boundary
    assert actors["source2"]["actor_id"] == "source2_actor"
    assert (
        actors["source2"]["planning_source_path_authority"]["source_actor_id"]
        == "native_static"
    )
    assert actors["source2"]["action_id_path"] == ["idle"] * 75
    assert actors["source2"]["action_phase_path"] == [0.0] * 75
    static_forward = actors["source2"]["anatomical_forward_habitat_world_path"][0]
    assert static_forward == pytest.approx(
        [-1.0 / (10.0**0.5), 0.0, -3.0 / (10.0**0.5)]
    )
    centers = source_center_paths(profile)
    assert centers["source1"][0] == pytest.approx([0.0, 1.97, 3.0])
    assert centers["source2"][0] == pytest.approx([2.0, 2.01, 3.0])
    assert profile["rir_expectation"]["requested_pair_state_count"] == 150
    assert profile["rir_expectation"]["unique_rir_job_count"] == 76
    target_event = profile["authorities"]["candidate"]["value"]["audio_program"][
        "target_event"
    ]
    assert target_event["start_sample"] == 7_467
    assert target_event["end_sample_exclusive"] == 53_379
    assert target_event["end_sample_exclusive"] - target_event["start_sample"] == 45_912


def test_planning_profile_does_not_reuse_animal_action_or_forward(tmp_path):
    manifest_path, _, row = _write_planning_fixture(tmp_path)
    baseline = build_actor_motion_profile_from_planning(
        planning_manifest_path=manifest_path,
        episode_id=row["episode_id"],
    )
    source_path = Path(
        row["target"]["motion_profile_authority"]["source_path"]["source_suite"]
    )
    suite = json.loads(source_path.read_text())
    frames = suite["scenarios"][0]["plan"]["frames"]
    for frame in frames:
        for state in frame["actor_states"]:
            state["action_id"] = "forged_animal_action"
            state["anatomical_forward_habitat_world"] = [1.0, 0.0, 0.0]
    source_path.write_text(json.dumps(suite))
    mutated = build_actor_motion_profile_from_planning(
        planning_manifest_path=manifest_path,
        episode_id=row["episode_id"],
    )
    assert mutated == baseline
    actors = mutated["authorities"]["candidate"]["value"]["actors"]
    assert actors["source1"]["action_id_path"] == ["walk"] * 75
    assert actors["source2"]["action_id_path"] == ["idle"] * 75


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        ("period", "copied runtime declaration drift"),
        ("blueprint", "copied runtime declaration drift"),
        ("emitter", "copied runtime declaration drift"),
        ("offset_space", "runtime/emitter declaration is incomplete"),
        ("scale", "runtime/emitter declaration is incomplete"),
        ("audio", "audio event authority drift"),
        ("actor", "source actor must resolve exactly once"),
        ("root", "root/source state drift"),
    ],
)
def test_planning_row_profile_fails_closed_on_authority_drift(
    tmp_path, mutation, match
):
    manifest_path, _, row = _write_planning_fixture(tmp_path)
    if mutation == "period":
        row["target"]["motion_profile_authority"]["runtime"][
            "walk_phase_period_frames"
        ] = 20
    elif mutation == "blueprint":
        row["target"]["motion_profile_authority"]["runtime"]["blueprint_class_path"] = (
            "/Game/Forged.BP_C"
        )
    elif mutation == "emitter":
        row["target"]["motion_profile_authority"]["runtime"]["emitter_offset_m"][1] = (
            9.0
        )
    elif mutation == "offset_space":
        row["target"]["motion_profile_authority"]["runtime"]["emitter_offset_space"] = (
            "actor_root"
        )
    elif mutation == "scale":
        row["target"]["motion_profile_authority"]["runtime"]["actor_scale"] = 0.0
    elif mutation == "audio":
        row["audio_program"]["target_event"]["start_sample"] = 7_595
    elif mutation == "actor":
        row["target"]["motion_profile_authority"]["source_path"]["source_actor_id"] = (
            "missing"
        )
        row["target"]["source_actor_id"] = "missing"
    elif mutation == "root":
        row["target"]["root_path_m"][3][0] += 0.1
    manifest_path.write_text(json.dumps({"episodes": [row]}))
    with pytest.raises(ActorMotionProfileError, match=match):
        build_actor_motion_profile_from_planning(
            planning_manifest_path=manifest_path,
            episode_id=row["episode_id"],
        )


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        ("missing", "missing or not regular"),
        ("schema", "semantic contract drift"),
        ("scale", "semantic contract drift"),
        ("armature", "semantic contract drift"),
        ("walking", "animation binding drift"),
    ],
)
def test_planning_profile_reloads_ue_import_semantics(
    tmp_path: Path, mutation: str, match: str
) -> None:
    planning, _, row = _write_planning_fixture(tmp_path)
    ref = row["target"]["motion_profile_authority"]["runtime"]["ue_import_manifest_ref"]
    manifest_path = Path(ref["path"])
    if mutation == "missing":
        manifest_path.unlink()
    else:
        manifest = json.loads(manifest_path.read_text())
        if mutation == "schema":
            manifest["schema"] = "rocketbox_native_ue_import_v3"
        elif mutation == "scale":
            manifest["runtime_contract"]["actor_scale"] = 1.1
        elif mutation == "armature":
            manifest["glb_contract"]["armature_scale"] = [0.01, 0.01, 0.01]
        else:
            manifest["content"]["animations"]["Walking"] += "_forged"
        manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ActorMotionProfileError, match=match):
        build_actor_motion_profile_from_planning(
            planning_manifest_path=planning,
            episode_id=row["episode_id"],
        )


def test_planning_profile_rejects_selected_human_without_import_authority(
    tmp_path: Path,
) -> None:
    planning, registry_path, row = _write_planning_fixture(tmp_path)
    registry = json.loads(registry_path.read_text())
    unsupported = copy.deepcopy(registry["assets"][0])
    unsupported["asset_id"] = "construction_human"
    unsupported["revision"] = "native_runtime_ue_v1"
    unsupported["runtime_backends"]["spear_unreal"].pop("actor_scale")
    unsupported["runtime_backends"]["spear_unreal"].pop("ue_import_manifest_ref")
    registry["assets"].append(unsupported)
    registry_path.write_text(json.dumps(registry))
    role = row["target"]
    role["runtime_asset_id"] = unsupported["asset_id"]
    role["runtime_revision"] = unsupported["revision"]
    runtime = role["motion_profile_authority"]["runtime"]
    runtime["asset_id"] = unsupported["asset_id"]
    runtime["asset_revision"] = unsupported["revision"]
    planning.write_text(json.dumps({"episodes": [row]}))
    with pytest.raises(ActorMotionProfileError, match="audited UE import identity"):
        build_actor_motion_profile_from_planning(
            planning_manifest_path=planning,
            episode_id=row["episode_id"],
        )

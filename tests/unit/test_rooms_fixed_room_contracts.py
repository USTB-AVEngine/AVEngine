from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest

from avengine.contracts.json_io import load_json
from avengine.timeline.audio_program import bind_audio_program_hash
from avengine.rooms.fixed_room_contracts import (
    SCENARIO_CONTRACT,
    validate_anchor_library,
    validate_room_capsule,
    validate_scenario_suite,
    validate_trajectory_template_set,
)


ROOT = Path(__file__).resolve().parents[2]


def _copy_contract(value: dict) -> dict:
    """Keep fixture construction readable without adding publication metadata."""

    return deepcopy(value)


def _status(state: str = "run") -> dict:
    return {"state": state}


def _room_registry() -> dict:
    return load_json(ROOT / "examples/registry/rooms/room_registry.json")


def _room_capsule() -> dict:
    return _copy_contract(
        {
            "schema": "avengine_m6x_room_capsule_v1",
            "room_capsule_id": "fixed_room_contract_fixture",
            "revision": "v1",
            "source_scene_provenance": {
                "provider": "SPEAR_Unreal",
                "scene_id": "apartment_0000",
                "upstream_role": "visual_and_scene_authoring_provenance",
            },
            "room_registry_ref": {
                "registry_id": "avengine_m6_representative_rooms_v1",
                "room_id": "legacy_ue_apartment_0000_v1",
                "room_revision": "real_surface_export_pending_portable_package_v1",
            },
            "resource_refs": [
                {
                    "role": "navmesh",
                    "resource_id": "legacy_navmesh",
                },
                {
                    "role": "scene_dataset",
                    "resource_id": "legacy_scene_dataset_config",
                },
                {
                    "role": "scene_instance",
                    "resource_id": "legacy_scene_instance_config",
                },
                {
                    "role": "visual_scene",
                    "resource_id": "legacy_real_surface",
                },
            ],
            "acoustic_package_ref": {
                "acoustic_representation_id": "legacy_real_surface_acoustic_v1",
                "resource_id": "legacy_real_surface_acoustic_package",
            },
            "qualification_report_id": "legacy_ue_apartment_m6_audit_v1",
            "runtime_backend": "habitat_sim_avengine",
            "formal_canary_output": "habitat_native",
            "operating_area": {
                "authority": "declared_habitat_navmesh",
                "floor_height_m": 0.271,
                "bounds_m": [
                    [-5.91, -0.01, -6.97],
                    [5.96, 5.78, 7.10],
                ],
                "placement_claim": "source_center_only",
            },
            "fixed_visual_object_set": {
                "authority_resource_id": "legacy_scene_instance_config",
                "mutation": "forbidden",
            },
            "visual_profile": {
                "profile_id": "legacy_apartment_habitat_fixed_v1",
                "lighting_setup_id": "legacy_apartment_0000",
                "fixed": True,
                "hbao": True,
            },
            "acoustic_material_status": "research_placeholder",
            "forbidden_zones": [
                "loaded_rigid_collision_obbs",
                "navmesh_non_navigable",
            ],
            "camera_listener_rig": {
                "rig_id": "camera_rig_0",
                "formal_view_id": "view0",
                "listener_id": "listener0",
                "logical_camera_count": 1,
                "listener_count": 1,
                "camera_listener_colocated": True,
                "camera_listener_cooriented": True,
                "sensor_modalities": ["rgb", "depth", "semantic"],
                "pose_anchor_ids": ["camera_default"],
            },
            "anchor_library_ref": {
                "anchor_library_id": "fixed_room_anchor_fixture",
                "revision": "v1",
            },
            "trajectory_template_set_ref": {
                "trajectory_template_set_id": "fixed_room_trajectory_fixture",
                "revision": "v1",
            },
            "fixed_layout_policy": {
                "room_object_mutation": "forbidden",
                "furniture_mutation": "forbidden",
                "automatic_furnishing": False,
                "layout_authority": "room_registry_scene_instance_resource",
            },
            "admission_state": "research",
            "execution_status": _status(),
        }
    )


def _anchor(
    anchor_id: str,
    position: list[float],
    *,
    sector: str,
    fov: str,
    acoustic: str,
    kind: str = "entity_spawn",
) -> dict:
    return {
        "anchor_id": anchor_id,
        "kind": kind,
        "position_m": position,
        "yaw_deg": 0.0,
        "listener_relative_sector": sector,
        "expected_camera_fov": fov,
        "expected_acoustic_path": acoustic,
        **({} if kind == "camera_listener_pose" else {"los_probe_height_m": 1.2}),
        "validation_status": _status("pass"),
    }


def _anchor_library() -> dict:
    anchors = [
        _anchor(
            "camera_default",
            [0.0, 1.6, 0.0],
            sector="not_applicable",
            fov="not_applicable",
            acoustic="not_applicable",
            kind="camera_listener_pose",
        ),
        _anchor(
            "front_left",
            [-1.5, 0.0, -2.0],
            sector="front_left",
            fov="in_fov",
            acoustic="los",
        ),
        _anchor(
            "front_right",
            [1.5, 0.0, -2.0],
            sector="front_right",
            fov="in_fov",
            acoustic="los",
        ),
        _anchor(
            "left_start",
            [-2.0, 0.0, -1.0],
            sector="left",
            fov="in_fov",
            acoustic="los",
        ),
        _anchor(
            "nlos_occluded",
            [2.5, 0.0, 1.0],
            sector="right",
            fov="out_of_fov",
            acoustic="nlos",
        ),
        _anchor(
            "rear_right",
            [1.0, 0.0, 2.0],
            sector="rear_right",
            fov="out_of_fov",
            acoustic="los",
        ),
        _anchor(
            "right_end",
            [2.0, 0.0, -1.0],
            sector="right",
            fov="in_fov",
            acoustic="los",
        ),
    ]
    return _copy_contract(
        {
            "schema": "avengine_m6x_anchor_library_v1",
            "anchor_library_id": "fixed_room_anchor_fixture",
            "revision": "v1",
            "room_capsule_ref": {
                "room_capsule_id": "fixed_room_contract_fixture",
                "revision": "v1",
            },
            "coordinate_frame": "avengine_world_right_handed_y_up_m",
            "anchors": anchors,
            "execution_status": _status(),
        }
    )


def _route(
    route_id: str,
    anchor_ids: list[str],
    interpolation: str,
    *,
    anchor_frame_indices: list[int] | None = None,
) -> dict:
    return {
        "route_id": route_id,
        "anchor_ids": anchor_ids,
        "anchor_frame_indices": (
            anchor_frame_indices
            if anchor_frame_indices is not None
            else ([0] if len(anchor_ids) == 1 else [0, 74])
        ),
        "interpolation": interpolation,
    }


def _trajectory_templates() -> dict:
    templates = [
        {
            "template_id": "linear_left_to_right",
            "kind": "linear",
            "heading_policy": "path_tangent_keep_last_on_hold",
            "routes": [
                _route(
                    "linear_left_to_right_route",
                    ["left_start", "right_end"],
                    "piecewise_linear",
                )
            ],
            "execution_status": _status(),
        },
        {
            "template_id": "static_front_left",
            "kind": "static",
            "heading_policy": "path_tangent_keep_last_on_hold",
            "routes": [_route("static_front_left_route", ["front_left"], "hold")],
            "execution_status": _status(),
        },
        {
            "template_id": "static_front_right",
            "kind": "static",
            "heading_policy": "path_tangent_keep_last_on_hold",
            "routes": [_route("static_front_right_route", ["front_right"], "hold")],
            "execution_status": _status(),
        },
        {
            "template_id": "static_nlos_occluded",
            "kind": "static",
            "heading_policy": "path_tangent_keep_last_on_hold",
            "routes": [_route("static_nlos_occluded_route", ["nlos_occluded"], "hold")],
            "execution_status": _status(),
        },
        {
            "template_id": "static_rear_right",
            "kind": "static",
            "heading_policy": "path_tangent_keep_last_on_hold",
            "routes": [_route("static_rear_right_route", ["rear_right"], "hold")],
            "execution_status": _status(),
        },
    ]
    return _copy_contract(
        {
            "schema": "avengine_m6x_trajectory_template_set_v1",
            "trajectory_template_set_id": "fixed_room_trajectory_fixture",
            "revision": "v1",
            "room_capsule_ref": {
                "room_capsule_id": "fixed_room_contract_fixture",
                "revision": "v1",
            },
            "anchor_library_ref": {
                "anchor_library_id": "fixed_room_anchor_fixture",
                "revision": "v1",
            },
            "frame_rate_hz": 15,
            "frame_count": 75,
            "templates": templates,
            "execution_status": _status(),
        }
    )


def _event(
    event_id: str,
    endpoint_id: str,
    start: int,
    end: int,
) -> dict:
    duration = end - start
    return {
        "event_id": event_id,
        "source_endpoint_id": endpoint_id,
        "sound_asset_id": "fixture_sound",
        "start_tick": start * 3,
        "end_tick_exclusive": end * 3,
        "start_sample": start,
        "end_sample_exclusive": end,
        "source_start_sample": 0,
        "source_end_sample_exclusive": duration,
        "linear_gain": 0.2,
        "fade_samples": 10,
        "normalization_policy": "use_sound_asset_policy",
        "render_source_stem": True,
    }


def _program(program_id: str, mode: str, events: list[dict]) -> dict:
    value = {
        "schema": "avengine_m6_audio_program_v1",
        "program_id": program_id,
        "revision": "v1",
        "mode": mode,
        "timeline": {
            "time_base_hz": 48000,
            "ticks_per_frame": 3200,
            "video_fps": 15,
            "frame_count": 75,
            "sample_rate_hz": 16000,
            "ticks_per_sample": 3,
            "sample_count": 80000,
        },
        "candidate_source_endpoint_ids": ["endpoint_0", "endpoint_1"],
        "events": sorted(
            events,
            key=lambda item: (
                item["start_sample"],
                item["source_endpoint_id"],
                item["event_id"],
            ),
        ),
        "source_specific_stems": True,
        "admission_state": "research",
    }
    if mode == "counterfactual_route_swap":
        value["counterfactual"] = {
            "operation": "swap_source_endpoint_routing",
            "variants": ["A", "B"],
            "reference_variant": "A",
            "mapped_variant": "B",
            "endpoint_permutation": {
                "endpoint_0": "endpoint_1",
                "endpoint_1": "endpoint_0",
            },
            "allowed_changed_fields": ["events[*].source_endpoint_id"],
        }
    return bind_audio_program_hash(value)


def _audio_programs() -> list[dict]:
    return [
        _program(
            "program_s0",
            "one_active_of_n",
            [_event("s0_event", "endpoint_0", 1000, 2000)],
        ),
        _program(
            "program_s1",
            "counterfactual_route_swap",
            [_event("s1_event", "endpoint_0", 1000, 2000)],
        ),
        _program(
            "program_s2",
            "one_active_of_n",
            [_event("s2_event", "endpoint_1", 1000, 2000)],
        ),
        _program("program_s2_silent", "silent_negative", []),
        _program(
            "program_s3",
            "intermittent_events",
            [
                _event("s3_event_0", "endpoint_0", 1000, 2000),
                _event("s3_event_1", "endpoint_0", 3000, 4000),
            ],
        ),
        _program(
            "program_s4",
            "simultaneous_subset",
            [
                _event("s4_event_0", "endpoint_0", 1000, 2500),
                _event("s4_event_1", "endpoint_1", 1000, 2500),
            ],
        ),
        _program(
            "program_s5",
            "sequential_sources",
            [
                _event("s5_event_0", "endpoint_0", 1000, 2000),
                _event("s5_event_1", "endpoint_1", 3000, 4000),
            ],
        ),
    ]


def _binding(
    endpoint: str,
    anchor: str,
    template: str,
    route: str,
) -> dict:
    return {
        "source_endpoint_id": endpoint,
        "entity_instance_id": f"entity_{endpoint[-1]}",
        "spawn_anchor_id": anchor,
        "trajectory_template_id": template,
        "trajectory_route_id": route,
    }


def _scenario(
    scenario_id: str,
    program_id: str,
    mode: str,
    bindings: list[dict],
) -> dict:
    return {
        "scenario_id": scenario_id,
        "purpose": SCENARIO_CONTRACT[scenario_id][0],
        "listener_anchor_id": "camera_default",
        "capture_frame_window": {
            "start_frame": 0,
            "end_frame_exclusive": 75,
        },
        "audio_program_ref": {
            "program_id": program_id,
            "revision": "v1",
            "expected_mode": mode,
        },
        "audio_variants": ["A", "B"] if scenario_id == "S1" else ["A"],
        "source_bindings": bindings,
        "execution_status": _status(),
    }


def _scenario_suite() -> dict:
    front_pair = [
        _binding(
            "endpoint_0",
            "front_left",
            "static_front_left",
            "static_front_left_route",
        ),
        _binding(
            "endpoint_1",
            "front_right",
            "static_front_right",
            "static_front_right_route",
        ),
    ]
    scenarios = [
        _scenario("S0", "program_s0", "one_active_of_n", deepcopy(front_pair)),
        _scenario(
            "S1",
            "program_s1",
            "counterfactual_route_swap",
            [
                deepcopy(front_pair[0]),
                _binding(
                    "endpoint_1",
                    "rear_right",
                    "static_rear_right",
                    "static_rear_right_route",
                ),
            ],
        ),
        _scenario("S2", "program_s2", "one_active_of_n", deepcopy(front_pair)),
        _scenario(
            "S3",
            "program_s3",
            "intermittent_events",
            [
                _binding(
                    "endpoint_0",
                    "left_start",
                    "linear_left_to_right",
                    "linear_left_to_right_route",
                ),
                deepcopy(front_pair[1]),
            ],
        ),
        _scenario("S4", "program_s4", "simultaneous_subset", deepcopy(front_pair)),
        _scenario(
            "S5",
            "program_s5",
            "sequential_sources",
            [
                deepcopy(front_pair[0]),
                _binding(
                    "endpoint_1",
                    "nlos_occluded",
                    "static_nlos_occluded",
                    "static_nlos_occluded_route",
                ),
            ],
        ),
    ]
    scenarios[2]["silent_negative_program_ref"] = {
        "program_id": "program_s2_silent",
        "revision": "v1",
        "expected_mode": "silent_negative",
    }
    return _copy_contract(
        {
            "schema": "avengine_m6x_scenario_suite_v1",
            "scenario_suite_id": "fixed_room_s0_s5_fixture",
            "revision": "v1",
            "room_capsule_ref": {
                "room_capsule_id": "fixed_room_contract_fixture",
                "revision": "v1",
            },
            "anchor_library_ref": {
                "anchor_library_id": "fixed_room_anchor_fixture",
                "revision": "v1",
            },
            "trajectory_template_set_ref": {
                "trajectory_template_set_id": "fixed_room_trajectory_fixture",
                "revision": "v1",
            },
            "authority_refs": {
                "audio_program_schema": "avengine_m6_audio_program_v1",
                "timeline_schema": "avengine_authoritative_timeline_v2",
                "legacy_flag_registry_id": "legacy_m5_1_source_event_flags",
                "legacy_flag_definition_revision": "m5_1_v1",
            },
            "fixed_room_policy": {
                "same_room_revision": True,
                "same_scene_instance": True,
                "same_acoustic_package": True,
                "furniture_mutation": "forbidden",
                "automatic_furnishing": False,
            },
            "scenarios": scenarios,
            "execution_status": _status(),
        }
    )


def _validate_all() -> tuple[list[str], list[str], list[str], list[str]]:
    room = _room_capsule()
    anchors = _anchor_library()
    trajectories = _trajectory_templates()
    suite = _scenario_suite()
    return (
        validate_room_capsule(room, room_registry=_room_registry()),
        validate_anchor_library(anchors, room_capsule=room),
        validate_trajectory_template_set(
            trajectories, anchor_library=anchors, room_capsule=room
        ),
        validate_scenario_suite(
            suite,
            room_capsule=room,
            anchor_library=anchors,
            trajectory_templates=trajectories,
            audio_programs=_audio_programs(),
        ),
    )


def test_complete_reference_only_contract_set_validates() -> None:
    assert _validate_all() == ([], [], [], [])


def test_room_capsule_resolves_existing_resources_and_rejects_dense_state() -> None:
    capsule = _room_capsule()
    assert validate_room_capsule(capsule, room_registry=_room_registry()) == []

    dense = deepcopy(capsule)
    dense["fixed_object_transforms"] = [[1.0, 0.0, 0.0]]
    dense = _copy_contract(dense)
    assert any("Additional properties" in item for item in validate_room_capsule(dense))

    unknown = deepcopy(capsule)
    unknown["resource_refs"][0]["resource_id"] = "missing_navmesh"
    unknown = _copy_contract(unknown)
    assert any(
        "does not resolve in room record" in item
        for item in validate_room_capsule(unknown, room_registry=_room_registry())
    )


def test_room_capsule_binds_visual_objects_and_source_center_obstacles() -> None:
    wrong_authority = deepcopy(_room_capsule())
    wrong_authority["fixed_visual_object_set"]["authority_resource_id"] = (
        "legacy_real_surface"
    )
    errors = validate_room_capsule(wrong_authority)
    assert any("scene_instance resource reference" in item for item in errors)

    reversed_bounds = deepcopy(_room_capsule())
    reversed_bounds["operating_area"]["bounds_m"][1][0] = -6.0
    errors = validate_room_capsule(reversed_bounds)
    assert any("increasing lower/upper bounds" in item for item in errors)

    missing_obstacle_authority = deepcopy(_room_capsule())
    missing_obstacle_authority["forbidden_zones"] = [
        "navmesh_non_navigable",
        "unrelated_zone",
    ]
    errors = validate_room_capsule(missing_obstacle_authority)
    assert any("loaded_rigid_collision_obbs" in item for item in errors)


def test_camera_anchor_does_not_carry_a_source_los_probe() -> None:
    anchors = deepcopy(_anchor_library())
    anchors["anchors"][0]["los_probe_height_m"] = 1.6
    errors = validate_anchor_library(anchors)
    assert any("cannot define los_probe_height_m" in item for item in errors)


def test_single_camera_listener_and_furniture_freeze_are_schema_invariants() -> None:
    capsule = deepcopy(_room_capsule())
    capsule["camera_listener_rig"]["logical_camera_count"] = 2
    capsule["fixed_layout_policy"]["furniture_mutation"] = "allowed"
    capsule = _copy_contract(capsule)
    errors = validate_room_capsule(capsule)
    assert any("logical_camera_count" in item for item in errors)
    assert any("furniture_mutation" in item for item in errors)


@pytest.mark.parametrize("state", ["run", "pass", "not_run"])
def test_status_vocabulary_accepts_nonblocked_states(state: str) -> None:
    capsule = deepcopy(_room_capsule())
    capsule["execution_status"] = {"state": state}
    capsule = _copy_contract(capsule)
    assert validate_room_capsule(capsule) == []


def test_blocked_status_requires_reason_and_evidence() -> None:
    anchors = deepcopy(_anchor_library())
    anchors["anchors"][1]["validation_status"] = {"state": "blocked"}
    anchors = _copy_contract(anchors)
    errors = validate_anchor_library(anchors)
    assert any("reason" in item for item in errors)
    assert any("evidence_refs" in item for item in errors)

    anchors["anchors"][1]["validation_status"] = {
        "state": "blocked",
        "reason": "native clearance evidence is unavailable",
        "evidence_refs": ["evidence://anchor/front_left/clearance"],
    }
    anchors = _copy_contract(anchors)
    assert validate_anchor_library(anchors) == []


def test_trajectory_templates_reference_sparse_anchors_and_reject_camera_routes() -> (
    None
):
    anchors = _anchor_library()
    trajectories = _trajectory_templates()
    assert validate_trajectory_template_set(trajectories, anchor_library=anchors) == []

    invalid = deepcopy(trajectories)
    invalid["templates"][0]["routes"][0]["anchor_ids"][0] = "camera_default"
    invalid = _copy_contract(invalid)
    assert any(
        "cannot use camera anchor" in item
        for item in validate_trajectory_template_set(invalid, anchor_library=anchors)
    )


def test_trajectory_frame_indices_align_and_stay_inside_master_capture() -> None:
    trajectories = deepcopy(_trajectory_templates())
    route = trajectories["templates"][0]["routes"][0]
    route["anchor_frame_indices"] = [1, 75]
    errors = validate_trajectory_template_set(trajectories)
    assert any("start at zero" in item for item in errors)


def test_s0_s5_modes_resolve_existing_audio_program_contracts() -> None:
    suite = _scenario_suite()
    observed = {
        item["scenario_id"]: (
            item["purpose"],
            item["audio_program_ref"]["expected_mode"],
        )
        for item in suite["scenarios"]
    }
    assert observed == SCENARIO_CONTRACT
    assert (
        validate_scenario_suite(
            suite,
            room_capsule=_room_capsule(),
            anchor_library=_anchor_library(),
            trajectory_templates=_trajectory_templates(),
            audio_programs=_audio_programs(),
        )
        == []
    )


def test_s2_requires_an_exact_zero_silent_negative_program() -> None:
    suite = deepcopy(_scenario_suite())
    suite["scenarios"][2]["silent_negative_program_ref"] = {
        "program_id": "program_s0",
        "revision": "v1",
        "expected_mode": "silent_negative",
    }
    errors = validate_scenario_suite(suite, audio_programs=_audio_programs())
    assert "S2 control must use silent_negative mode" in errors


def test_capture_window_and_audio_program_frame_counts_match() -> None:
    suite = deepcopy(_scenario_suite())
    suite["scenarios"][0]["capture_frame_window"] = {
        "start_frame": 0,
        "end_frame_exclusive": 74,
    }
    errors = validate_scenario_suite(
        suite,
        trajectory_templates=_trajectory_templates(),
        audio_programs=_audio_programs(),
    )
    assert "S0 AudioProgram frame_count differs from capture window" in errors


def test_two_overlapping_of_n_is_not_a_parallel_audio_program_mode() -> None:
    suite = deepcopy(_scenario_suite())
    suite["scenarios"][4]["audio_program_ref"]["expected_mode"] = "two_overlapping_of_n"
    suite = _copy_contract(suite)
    errors = validate_scenario_suite(suite)
    assert any("two_overlapping_of_n" in item for item in errors)

    audio_program_schema = load_json(ROOT / "schemas/m6_audio_program_v1.schema.json")
    assert (
        "two_overlapping_of_n" not in audio_program_schema["properties"]["mode"]["enum"]
    )


def test_suite_references_timeline_and_flags_without_embedding_replacements() -> None:
    suite = deepcopy(_scenario_suite())
    suite["timeline"] = {"frames": []}
    suite["flags"] = {"stationary": True}
    suite = _copy_contract(suite)
    errors = validate_scenario_suite(suite)
    assert any("Additional properties" in item for item in errors)

    valid = _scenario_suite()
    assert valid["authority_refs"] == {
        "audio_program_schema": "avengine_m6_audio_program_v1",
        "timeline_schema": "avengine_authoritative_timeline_v2",
        "legacy_flag_registry_id": "legacy_m5_1_source_event_flags",
        "legacy_flag_definition_revision": "m5_1_v1",
    }


def test_scenario_anchor_semantics_fail_closed() -> None:
    anchors = deepcopy(_anchor_library())
    nlos = next(
        item for item in anchors["anchors"] if item["anchor_id"] == "nlos_occluded"
    )
    nlos["expected_acoustic_path"] = "los"
    anchors = _copy_contract(anchors)
    errors = validate_scenario_suite(
        _scenario_suite(),
        anchor_library=anchors,
        trajectory_templates=_trajectory_templates(),
    )
    assert "S5 requires both LOS and NLOS anchors" in errors


def test_missing_or_duplicate_scenario_ids_report_errors_without_keyerror() -> None:
    duplicate = deepcopy(_scenario_suite())
    duplicate["scenarios"][5]["scenario_id"] = "S4"
    duplicate["scenarios"][5]["purpose"] = "overlapping_sources"
    duplicate["scenarios"][5]["audio_program_ref"]["expected_mode"] = (
        "simultaneous_subset"
    )
    errors = validate_scenario_suite(
        duplicate,
        anchor_library=_anchor_library(),
        trajectory_templates=_trajectory_templates(),
    )
    assert "scenarios must contain canonical S0..S5 order exactly once" in errors


def test_git_tracked_configs_do_not_require_manual_content_hashes() -> None:
    suite = _scenario_suite()
    suite["revision"] = "tampered"
    assert "content_sha256" not in suite
    assert validate_scenario_suite(suite) == []

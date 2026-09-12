from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path

from avengine.dataset import production_runner as runner


def _registry():
    return {
        "assets": [
            {"asset_id": "human", "entity_class": "articulated_human"},
            {"asset_id": "human2", "entity_class": "articulated_human"},
            {"asset_id": "animal", "entity_class": "articulated_animal"},
            {"asset_id": "device", "entity_class": "rigid_static_object"},
        ]
    }


def _ordinary(request_id, assets, targets):
    return {
        "request_id": request_id,
        "room_id": "room_apartment",
        "instances": [
            {"asset_id": asset, "instance_id": f"source{index + 1}",
             "source_class": source}
            for index, (asset, source) in enumerate(assets)
        ],
        "qa_targets": targets,
        "profile": {"retry_budget_within_profile": 197},
    }


def _core():
    members = []
    for role, order in (
        ("v0_a0", ("human", "animal")),
        ("v0_a1", ("human", "animal")),
        ("v1_a0", ("animal", "human")),
        ("v1_a1", ("animal", "human")),
    ):
        members.append({
            "request_id": f"core_template_{role}",
            "member_role": role,
            "instances": [
                {"asset_id": asset, "instance_id": f"source{index + 1}",
                 "source_class": (
                     "articulated_human" if asset == "human"
                     else "articulated_animal"
                 )}
                for index, asset in enumerate(order)
            ],
        })
    return {
        "group_id": "core_template",
        "task_family": "visible_binding",
        "room_id": "room_apartment",
        "members": members,
        "shared_audio_member_ids": [
            ["core_template_v0_a0", "core_template_v1_a0"],
            ["core_template_v0_a1", "core_template_v1_a1"],
        ],
        "shared_visual_member_ids": [
            ["core_template_v0_a0", "core_template_v0_a1"],
            ["core_template_v1_a0", "core_template_v1_a1"],
        ],
    }


def _inputs(config, *, ordinary=2, core=1):
    return {
        "config": config,
        "registry": _registry(),
        "room_catalog": {"rooms": [
            {"room_id": "room_apartment", "room_family": "apartment"},
            {"room_id": "room_hm3d", "room_family": "hm3d"},
        ]},
        "max_new_ordinary": ordinary,
        "max_new_core_groups": core,
        "max_planned_requests": ordinary + core,
        "round_limit": 4,
    }


def test_p19_planner_uses_registered_combo_room_and_exact_qa_target():
    config = {
        "schema": "avengine_v1_production_spec_v1",
        "batch_id": "planner_test",
        "seed": 10,
        "defaults": {},
        "episodes": [
            _ordinary(
                "human_animal_template",
                (("human", "articulated_human"), ("animal", "articulated_animal")),
                [{"qa_id": "QA-06", "branch": "moving",
                  "target_instance_ids": ["source2"],
                  "event": {"kind": "target_audible_window"}}],
            ),
        ],
        "core_groups": [_core()],
    }
    feedback = {
        "outstanding_requests": [
            {"kind": "core_group_cell", "core_task_family": "visible_binding",
             "room_family": "apartment", "remaining_group_count": 1},
            {"kind": "qa_branch", "qa_id": "QA-06", "branch": "moving",
             "entity_combination": "human+animal"},
        ]
    }
    result = runner.plan_v1_next_requests(
        manifest={"episodes": [], "production": {"core_groups": []}},
        feedback=feedback,
        coverage_inputs=_inputs(config),
    )
    assert result["status"] == "planned"
    assert [row["kind"] for row in result["requests"]] == [
        "core_group", "ordinary_episode"
    ]
    ordinary = result["requests"][1]
    assert ordinary["request"]["request_id"].endswith("backfill_002")
    assert ordinary["request"].get("qa_targets") is None
    assert ordinary["target"]["target_instance_ids"] == ["source2"]
    assert ordinary["planning"]["retry_budget_within_profile"] == 197
    group = result["requests"][0]["config_block"]
    assert group["room_id"] == "room_apartment"
    member_ids = {
        member["request_id"] for member in group["members"]
    }
    assert all(
        value in member_ids
        for pair in group["shared_audio_member_ids"]
        for value in pair
    )


def test_p19_planner_blocks_unmatched_registered_combination():
    config = {
        "schema": "avengine_v1_production_spec_v1",
        "batch_id": "planner_test",
        "seed": 10,
        "defaults": {},
        "episodes": [
            _ordinary(
                "human_animal_template",
                (("human", "articulated_human"), ("animal", "articulated_animal")),
                [],
            ),
        ],
        "core_groups": [],
    }
    result = runner.plan_v1_next_requests(
        manifest={"episodes": [], "production": {"core_groups": []}},
        feedback={"outstanding_requests": [{
            "kind": "entity_combination",
            "entity_combination": "animal+device",
            "remaining_distinct_worlds": 1,
        }]},
        coverage_inputs=_inputs(config, ordinary=2, core=0),
    )
    assert result["status"] == "blocked"
    assert result["requests"] == []
    assert "no ordinary template" in result["skipped"][0]["reason"]


def test_p19_ordinary_cap_is_global_over_manifest_resume_snapshot():
    config = {
        "schema": "avengine_v1_production_spec_v1",
        "batch_id": "planner_test",
        "seed": 10,
        "defaults": {},
        "episodes": [
            _ordinary(
                "human_animal_template",
                (("human", "articulated_human"), ("animal", "articulated_animal")),
                [],
            ),
        ],
        "core_groups": [],
    }
    manifest = {
        "episodes": [{
            "episode_id": "already_backfilled",
            "p19_planned": {"round": 1},
        }],
        "production": {"core_groups": []},
    }
    result = runner.plan_v1_next_requests(
        manifest=manifest,
        feedback={"outstanding_requests": [{
            "kind": "qa_id", "qa_id": "QA-06",
            "remaining_valid_main_questions": 1,
        }]},
        coverage_inputs=_inputs(config, ordinary=1, core=0),
    )
    assert result["requests"] == []


def test_manifest_resource_policy_is_consumed_and_cli_parallel_reaches_runner(
        monkeypatch, tmp_path):
    policy = runner.ResourcePolicy(
        backends=runner.backend_profiles(None)
    ).to_dict()
    policy["cpu"]["max_workers"] = 2
    policy["gpu"]["max_workers"] = 1
    policy["gpu"]["max_workers_per_device"] = 1
    manifest = {
        "schema": "avengine_qa_batch_manifest_v1",
        "batch_id": "resource_policy_test",
        "episodes": [],
        "production": {"core_groups": []},
        "resource_policy": policy,
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    instance = runner.ProductionRunner(
        manifest=manifest,
        manifest_path=manifest_path,
        run_root=tmp_path / "run",
        max_parallel=3,
    )
    assert instance.max_parallel == 3
    assert instance.resource_policy_source == "manifest"
    assert instance.broker.allocator.policy.cpu.max_workers == 2
    assert instance.broker.allocator.policy.gpu.max_workers == 1
    saved = instance.state()
    assert saved["resource_policy_source"] == "manifest"
    assert saved["resource_policy"]["cpu"]["max_workers"] == 2

    spec = importlib.util.spec_from_file_location(
        "qa_batch_runner_t09_test",
        Path(__file__).resolve().parents[2] / "tools/dataset/run_qa_batch.py",
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    seen = {}

    def fake_run_production(**kwargs):
        seen.update(kwargs)
        return {
            "status": "complete",
            "run_root": str(tmp_path / "run"),
            "run_summary": {
                "delivered_groups": [], "delivered_episodes": [],
                "native_visual_worlds_used": 0,
            },
            "coverage_feedback": {"deficits": []},
            "delivery_export": None,
        }

    monkeypatch.setattr(runner, "run_production", fake_run_production)
    module.execute_production(
        manifest_path,
        tmp_path / "run",
        max_parallel=3,
    )
    assert seen["max_parallel"] == 3


def test_p19_export_does_not_turn_retained_world_into_fresh(tmp_path):
    receipt = tmp_path / "capture_receipt.json"
    receipt.write_text("{}", encoding="utf-8")
    cached = {
        "source_kind": "retained_group_library_census",
        "members": [{"world_id": "world_old", "group_id": "g_old"}],
    }
    exported = {
        "source_kind": "delivered_export",
        "members": [
            {"world_id": "world_old", "group_id": "g_old"},
            {"world_id": "world_new", "group_id": "g_new"},
        ],
    }
    unknown = runner._p19_world_origins([cached, exported])
    assert unknown["fresh_world_count"] == 0
    assert unknown["unknown_world_ids"] == ["world_new"]
    known = runner._p19_world_origins(
        [cached, exported], declared_fresh_world_ids=["world_new"]
    )
    assert known["retained_world_count"] == 1
    assert known["fresh_world_count"] == 1
    by_world = {row["world_id"]: row["origin"] for row in known["worlds"]}
    assert by_world == {"world_old": "retained", "world_new": "fresh"}


def test_fresh_file_exists_is_not_capture_provenance(tmp_path):
    receipt = tmp_path / "not_a_capture_receipt.txt"
    receipt.write_text("not a capture receipt", encoding="utf-8")
    result = runner._p19_fresh_world_declarations({
        "fresh_world_ids": ["negative_fixture_world"],
        "fresh_world_evidence": {
            "negative_fixture_world": {
                "capture_receipt_path": str(receipt)
            }
        },
    })
    assert result["validated_world_ids"] == []
    assert "negative_fixture_world" in result["invalid"]


def test_identity_core_template_rejects_static_device_and_accepts_humans():
    bad = _core()
    bad["group_id"] = "identity_bad"
    bad["task_family"] = "cross_event_identity"
    for member in bad["members"]:
        member["instances"][1] = {
            "asset_id": "device",
            "instance_id": "source2",
            "source_class": "rigid_static_object",
        }
    good = copy.deepcopy(bad)
    good["group_id"] = "identity_good"
    for member in good["members"]:
        member["instances"] = [
            {
                "asset_id": "human",
                "instance_id": "source1",
                "source_class": "articulated_human",
            },
            {
                "asset_id": "human2",
                "instance_id": "source2",
                "source_class": "articulated_human",
            },
        ]
    config = {
        "schema": "avengine_v1_production_spec_v1",
        "batch_id": "identity_legality",
        "seed": 4,
        "defaults": {},
        "episodes": [],
        "core_groups": [bad, good],
    }
    feedback = {
        "outstanding_requests": [{
            "kind": "fresh_world",
            "core_task_family": "cross_event_identity",
            "room_family": "apartment",
            "world_id": "fresh_identity_01",
        }]
    }
    result = runner.plan_v1_next_requests(
        manifest={"episodes": [], "production": {"core_groups": []}},
        feedback=feedback,
        coverage_inputs={
            **_inputs(config, ordinary=0, core=1),
            "max_planned_requests": 1,
        },
    )
    assert result["status"] == "planned"
    assert result["requests"][0]["config_block"]["group_id"].startswith("identity_good")
    assert result["skipped"] == []
    bad_only = {**config, "core_groups": [bad]}
    blocked = runner.plan_v1_next_requests(
        manifest={"episodes": [], "production": {"core_groups": []}},
        feedback=feedback,
        coverage_inputs={
            **_inputs(bad_only, ordinary=0, core=1),
            "max_planned_requests": 1,
        },
    )
    assert blocked["requests"] == []
    assert "articulated_human" in blocked["skipped"][0]["reason"]


# --------------------------------------------------------------------------
# C02-R1: the UE/SPEAR producer's own receipt shape
#
# The retained UE capture writes no `schema` key and reports status
# "research_only", so the old validator -- which keyed on an
# `avengine_m5_1_` prefix and status "pass" -- matched no real UE output at
# all. It is recognised by the shape it actually writes, and then checked as
# strictly as the Habitat path. Recognising the structure is not a birth:
# a fresh world still needs a stage run that binds it.
# --------------------------------------------------------------------------

import pytest

from avengine.dataset.source_capabilities import (
    STATE_NOT_APPLICABLE as NOT_APPLICABLE,
)

UE_SIBLINGS = (
    "neutral_readback.json",
    "frame_readbacks.json",
    "native_runtime_binding_readback.json",
    "native_pixel_runtime_readbacks.json",
    "pixel_visibility_truth.json",
)


def _ue_receipt(capture_root):
    """The shape the real producer writes, trimmed to what is checked."""
    return {
        "backend_role": "production_visual",
        "status": "research_only",
        "research_only": True,
        "qualification": False,
        "qualification_claim": False,
        "episode_counted": False,
        "formal_dataset_count": 0,
        "clock": {"frame_count": 150, "frame_rate_hz": 15, "ticks_per_frame": 3200},
        "audio": {"status": "not_requested"},
        "rlr": {"status": "not_requested"},
        "scene": {
            "map_path": "/Game/AVEngine/Optional/Kujiale/kujiale_0020_full_home_v1",
            "map_path_status": "launched",
            "room_id": "kujiale_0020_full_home_v1",
            "scene_id": "kujiale_0020_full_home_v1",
        },
        "native_level_readback": {
            "expected": "kujiale_0020_full_home_v1",
            "observed": "kujiale_0020_full_home_v1",
            "method": "UGameplayStatics.GetCurrentLevelName",
            "status": "pass",
        },
        "native_pixel": {
            "status": "pass",
            "artifacts": {
                "metric_depth": str(capture_root / "metric_depth_native.npz"),
            },
        },
        "media": {
            "ue_visual_only": {
                "path": str(capture_root / "ue_visual_only.mp4"),
                "frame_count": 150,
                "frame_rate_hz": 15,
                "duration_seconds": 10.0,
                "status": "pass",
            }
        },
    }


def _ue_capture_tree(tmp_path, *, bind_world=True, mutate=None, drop_siblings=()):
    group_root = tmp_path / "stage_run" / "ue_group_01"
    capture_root = group_root / "v0_capture" / "capture" / "attempt_01" / "capture"
    capture_root.mkdir(parents=True)
    (capture_root / "metric_depth_native.npz").write_bytes(b"depth")
    (capture_root / "ue_visual_only.mp4").write_bytes(b"video")
    for name in UE_SIBLINGS:
        if name in drop_siblings:
            continue
        (capture_root / name).write_text("{}", encoding="utf-8")
    receipt = _ue_receipt(capture_root)
    if mutate is not None:
        mutate(receipt)
    (capture_root / "research_receipt.json").write_text(
        json.dumps(receipt), encoding="utf-8")
    request = capture_root.parent / "request.json"
    request.write_text(json.dumps({
        "room_id": "kujiale_0020_full_home_v1",
        "task_family": "reference_expansion",
        "frame_count": 150, "frame_rate_hz": 15,
    }), encoding="utf-8")
    if bind_world:
        (tmp_path / "stage_run" / "ue_group_01_stage_run_001.json").write_text(
            json.dumps({
                "schema": "avengine_native_group_stage_run_v1",
                "world_id": "world_ue_group_01",
                "room_id": "kujiale_0020_full_home_v1",
                "task_family": "reference_expansion",
                "output_root": str(group_root.resolve()),
                "executed": [{"work_item_id": "ue_group_01/v0_capture:capture:01",
                              "status": "pass"}],
            }), encoding="utf-8")
    return {
        "capture_receipt_path": str(capture_root / "research_receipt.json"),
        "request_path": str(request),
        "expected_room_id": "kujiale_0020_full_home_v1",
        "expected_task_family": "reference_expansion",
    }


def _declare(entry, world_id="world_ue_group_01"):
    return runner._p19_fresh_world_declarations({
        "fresh_world_ids": [world_id],
        "fresh_world_evidence": {world_id: entry},
    })


def test_a_ue_receipt_without_a_schema_key_is_still_recognised(tmp_path):
    entry = _ue_capture_tree(tmp_path)
    result = _declare(entry)
    assert result["validated_world_ids"] == ["world_ue_group_01"], result["invalid"]
    detail = result["validation"]["world_ue_group_01"]
    assert detail["capture_backend"] == "ue_native"
    assert detail["ue_level_loaded"] == "kujiale_0020_full_home_v1"
    assert detail["ue_backend_role"] == "production_visual"


def test_a_visual_only_capture_needs_no_sample_clock(tmp_path):
    """Audio is declared not_requested; demanding a sample rate rejected it."""
    entry = _ue_capture_tree(tmp_path)
    detail = _declare(entry)["validation"]["world_ue_group_01"]
    assert detail["clock"]["frame_count"] == 150
    assert detail["clock"]["sample_count"] is None
    assert detail["clock"]["audio_requested"] is False


def test_a_ue_capture_that_claims_audio_must_carry_its_sample_clock(tmp_path):
    entry = _ue_capture_tree(
        tmp_path,
        mutate=lambda value: value.__setitem__("audio", {"status": "rendered"}),
    )
    result = _declare(entry)
    assert result["validated_world_ids"] == []
    assert "clock is incomplete" in result["invalid"]["world_ue_group_01"]


def test_a_structurally_sound_ue_receipt_is_not_by_itself_a_fresh_birth(tmp_path):
    """No stage run binds a world, so nothing was born here."""
    entry = _ue_capture_tree(tmp_path, bind_world=False)
    result = _declare(entry)
    assert result["validated_world_ids"] == []
    assert "does not bind this world" in result["invalid"]["world_ue_group_01"]


@pytest.mark.parametrize(
    "label,mutate,drop,expected",
    [
        ("comparison_visual",
         lambda v: v.__setitem__("backend_role", "comparison_visual"), (),
         "not production visual output"),
        ("level_not_launched",
         lambda v: v["scene"].__setitem__("map_path_status", "requested"), (),
         "does not prove the level launched"),
        ("loaded_another_level",
         lambda v: v["native_level_readback"].__setitem__("observed", "other_map"),
         (), "is not the level requested"),
        ("scene_and_level_disagree",
         lambda v: v["scene"].__setitem__("room_id", "apartment_0000"), (),
         "name different rooms"),
        ("no_pixel_artifact",
         lambda v: v["native_pixel"].__setitem__("artifacts", {}), (),
         "no readable pixel artifact"),
        ("pixel_artifact_absent_from_disk",
         lambda v: v["native_pixel"]["artifacts"].__setitem__(
             "metric_depth", "/nowhere/metric_depth_native.npz"), (),
         "no readable pixel artifact"),
        ("media_clock_disagrees",
         lambda v: v["media"]["ue_visual_only"].__setitem__("frame_count", 90), (),
         "media clock differs"),
        ("counted_as_an_episode",
         lambda v: v.__setitem__("episode_counted", True), (),
         "already counted as an Episode"),
        ("not_research_only",
         lambda v: v.__setitem__("research_only", False), (),
         "not a research-only capture"),
        ("claims_qualification",
         lambda v: v.__setitem__("qualification_claim", True), (),
         "not a research-only capture"),
        ("execution_readbacks_missing", None,
         ("pixel_visibility_truth.json",),
         "execution readbacks are not beside"),
    ],
)
def test_one_thing_wrong_refuses_the_ue_capture(tmp_path, label, mutate, drop, expected):
    entry = _ue_capture_tree(tmp_path, mutate=mutate, drop_siblings=drop)
    result = _declare(entry)
    assert result["validated_world_ids"] == [], label
    assert expected in result["invalid"]["world_ue_group_01"], (
        label, result["invalid"])


def test_a_document_that_only_says_pass_is_not_a_ue_capture(tmp_path):
    receipt = tmp_path / "research_receipt.json"
    receipt.write_text(json.dumps({"status": "pass", "research_only": True}),
                       encoding="utf-8")
    result = _declare({"capture_receipt_path": str(receipt)})
    assert result["validated_world_ids"] == []
    assert "schema is unsupported" in result["invalid"]["world_ue_group_01"]


def test_an_empty_artifact_map_is_not_a_ue_capture(tmp_path):
    entry = _ue_capture_tree(
        tmp_path, mutate=lambda v: v.__setitem__("media", {}))
    result = _declare(entry)
    assert result["validated_world_ids"] == []
    assert "no readable media" in result["invalid"]["world_ue_group_01"]


# --------------------------------------------------------------------------
# C02-R1: an asset deficit has to reach the request
#
# A source_fine_type deficit names no entity combination, so the planner took
# episodes[0] and appended a request that never mentioned the missing asset.
# A qa_id deficit with no branch fell through to "no target", and the request
# went out with the config's default coverage instead of the question it was
# appended to answer. Both produced requests that could not close the deficit
# they were raised for.
# --------------------------------------------------------------------------


def _asset_registry():
    return {
        "assets": [
            {"asset_id": "human", "entity_class": "articulated_human",
             "runtime_backends": {"spear_unreal": {}, "habitat": {}}},
            {"asset_id": "dog", "entity_class": "articulated_animal",
             "runtime_backends": {"spear_unreal": {}, "habitat": {}}},
            {"asset_id": "cat", "entity_class": "articulated_animal",
             "runtime_backends": {"spear_unreal": {}, "habitat": {}}},
            {"asset_id": "ue_only_lamp", "entity_class": "rigid_static_object",
             "runtime_backends": {"spear_unreal": {}}},
            {"asset_id": "speaker", "entity_class": "rigid_static_object",
             "runtime_backends": {"spear_unreal": {}, "habitat": {}}},
        ]
    }


def _asset_catalog():
    return {"rooms": [
        {"room_id": "room_apartment", "room_family": "apartment",
         "renderer": "ue_spear"},
        {"room_id": "room_hm3d", "room_family": "hm3d", "renderer": "habitat"},
    ]}


def _episode(request_id, room_id, pairs, targets):
    return {
        "request_id": request_id,
        "room_id": room_id,
        "instances": [
            {"asset_id": asset, "instance_id": f"source{index + 1}",
             "source_class": source}
            for index, (asset, source) in enumerate(pairs)
        ],
        "qa_targets": targets,
        "profile": {"retry_budget_within_profile": 8},
    }


ANIMAL = "articulated_animal"
HUMAN = "articulated_human"
DEVICE = "rigid_static_object"


def _asset_config(episodes):
    return {
        "schema": "avengine_v1_production_spec_v1",
        "batch_id": "asset_planner_test",
        "seed": 3,
        "episodes": episodes,
        "core_groups": [],
    }


def _plan(config, outstanding, *, ordinary=8, registry=None, catalog=None):
    inputs = {
        "config": config,
        "registry": registry or _asset_registry(),
        "room_catalog": catalog or _asset_catalog(),
        "max_new_ordinary": ordinary,
        "max_new_core_groups": 0,
        "max_planned_requests": ordinary,
    }
    return runner.plan_v1_next_requests(
        manifest={"episodes": [], "production": {}},
        feedback={"outstanding_requests": outstanding},
        coverage_inputs=inputs,
    )


def _fine_type_row(fine_type, untested, **extra):
    return {
        "kind": "source_fine_type",
        "fine_type": fine_type,
        "state": "evidence_missing_or_unsampled",
        "candidate_asset_ids": list(untested),
        "untested_asset_ids": list(untested),
        "refused_asset_ids": [],
        "blocking_dimensions": ["clearance"],
        **extra,
    }


def test_an_asset_deficit_puts_that_asset_into_the_request():
    config = _asset_config([
        _episode("apartment_human_dog", "room_apartment",
                 [("human", HUMAN), ("dog", ANIMAL)],
                 [{"qa_id": "QA-01", "target_instance_ids": ["source1"]}]),
    ])
    plan = _plan(config, [_fine_type_row("animal/cat/abyssinian", ["cat"])])
    assert plan["added_request_count"] == 1, plan.get("skipped")
    entry = plan["requests"][0]
    binding = entry["source_type_binding"]
    assert binding["bound_asset_id"] == "cat"
    assert binding["replaced_asset_id"] == "dog"
    assert binding["source_family"] == "animal"
    assert {item["asset_id"] for item in entry["request"]["instances"]} == {
        "human", "cat"}
    # The request still asks something, so the eight dimensions have
    # something to be measured against.
    assert entry["request"]["qa_targets"]


def test_the_asset_never_lands_in_a_slot_of_another_family():
    config = _asset_config([
        _episode("apartment_human_dog", "room_apartment",
                 [("human", HUMAN), ("dog", ANIMAL)],
                 [{"qa_id": "QA-01", "target_instance_ids": ["source1"]}]),
    ])
    plan = _plan(config, [_fine_type_row("device/speaker", ["speaker"])])
    assert plan["added_request_count"] == 0
    assert "no instance slot of family 'device'" in plan["skipped"][0]["reason"]


def test_a_room_the_asset_was_not_built_for_is_not_chosen():
    config = _asset_config([
        _episode("hm3d_human_dog", "room_hm3d",
                 [("human", HUMAN), ("dog", ANIMAL)],
                 [{"qa_id": "QA-01", "target_instance_ids": ["source1"]}]),
        _episode("apartment_human_lamp", "room_apartment",
                 [("human", HUMAN), ("speaker", DEVICE)],
                 [{"qa_id": "QA-01", "target_instance_ids": ["source1"]}]),
    ])
    plan = _plan(config, [_fine_type_row("device/lamp", ["ue_only_lamp"])])
    assert plan["added_request_count"] == 1, plan.get("skipped")
    binding = plan["requests"][0]["source_type_binding"]
    # ue_only_lamp declares spear_unreal only, so the habitat room is out.
    assert binding["room_id"] == "room_apartment"
    assert binding["runtime_backend"] == "spear_unreal"


def test_a_refused_candidate_rotates_to_the_next_one():
    config = _asset_config([
        _episode("apartment_human_dog", "room_apartment",
                 [("human", HUMAN), ("dog", ANIMAL)],
                 [{"qa_id": "QA-01", "target_instance_ids": ["source1"]}]),
    ])
    row = _fine_type_row("animal/cat/any", [])
    row["candidate_asset_ids"] = ["ue_only_lamp", "cat"]
    row["refused_asset_ids"] = ["ue_only_lamp"]
    row["untested_asset_ids"] = ["cat"]
    plan = _plan(config, [row])
    assert plan["added_request_count"] == 1, plan.get("skipped")
    binding = plan["requests"][0]["source_type_binding"]
    assert binding["bound_asset_id"] == "cat"
    assert binding["candidate_rotation_index"] == 0


def test_an_asset_already_in_every_episode_needs_measurement_not_a_request():
    config = _asset_config([
        _episode("apartment_human_dog", "room_apartment",
                 [("human", HUMAN), ("dog", ANIMAL)],
                 [{"qa_id": "QA-01", "target_instance_ids": ["source1"]}]),
    ])
    plan = _plan(config, [_fine_type_row("animal/dog/border_collie", ["dog"])])
    assert plan["added_request_count"] == 0
    reason = plan["skipped"][0]["reason"]
    assert "already placed in a configured episode" in reason
    assert "needs the missing dimensions measured" in reason
    assert "clearance" in reason


def test_a_template_that_asks_nothing_is_not_used_to_carry_an_asset():
    config = _asset_config([
        _episode("silent_template", "room_apartment",
                 [("human", HUMAN), ("dog", ANIMAL)], []),
    ])
    plan = _plan(config, [_fine_type_row("animal/cat/abyssinian", ["cat"])])
    assert plan["added_request_count"] == 0
    assert "registers no qa_targets" in plan["skipped"][0]["reason"]


def test_a_qa_deficit_gets_a_template_that_registers_that_question():
    config = _asset_config([
        _episode("first_no_target", "room_apartment",
                 [("human", HUMAN), ("dog", ANIMAL)],
                 [{"qa_id": "QA-01", "target_instance_ids": ["source1"]}]),
        _episode("carries_qa06_still", "room_apartment",
                 [("human", HUMAN), ("dog", ANIMAL)],
                 [{"qa_id": "QA-06", "branch": "still",
                   "target_instance_ids": ["source1"]}]),
    ])
    plan = _plan(config, [
        {"kind": "qa_branch", "qa_id": "QA-06", "branch": "still",
         "state": "evidence_missing_or_unsampled"}])
    assert plan["added_request_count"] == 1, plan.get("skipped")
    entry = plan["requests"][0]
    assert entry["request"]["request_id"].startswith("carries_qa06_still")
    assert entry["target"]["qa_id"] == "QA-06"
    assert entry["target"]["branch"] == "still"


def test_a_qa_deficit_no_template_registers_is_skipped_not_sent_bare():
    config = _asset_config([
        _episode("only_qa01", "room_apartment",
                 [("human", HUMAN), ("dog", ANIMAL)],
                 [{"qa_id": "QA-01", "target_instance_ids": ["source1"]}]),
    ])
    plan = _plan(config, [
        {"kind": "qa_id", "qa_id": "QA-09",
         "state": "evidence_missing_or_unsampled"}])
    assert plan["added_request_count"] == 0
    assert "no registered target for QA-09" in plan["skipped"][0]["reason"]


def test_every_deficit_kind_gets_a_turn_inside_one_round_cap():
    """Asset deficits are appended last; list order starved them every round."""
    config = _asset_config([
        _episode("apartment_human_dog", "room_apartment",
                 [("human", HUMAN), ("dog", ANIMAL)],
                 [{"qa_id": "QA-01", "target_instance_ids": ["source1"]},
                  {"qa_id": "QA-02", "target_instance_ids": ["source1"]},
                  {"qa_id": "QA-03", "target_instance_ids": ["source1"]}]),
    ])
    outstanding = [
        {"kind": "qa_id", "qa_id": "QA-01", "state": "evidence_missing_or_unsampled"},
        {"kind": "qa_id", "qa_id": "QA-02", "state": "evidence_missing_or_unsampled"},
        {"kind": "qa_id", "qa_id": "QA-03", "state": "evidence_missing_or_unsampled"},
        _fine_type_row("animal/cat/abyssinian", ["cat"]),
    ]
    plan = _plan(config, outstanding, ordinary=2)
    kinds = {entry["reason"]["kind"] for entry in plan["requests"]}
    assert plan["added_request_count"] == 2
    assert kinds == {"qa_id", "source_fine_type"}


def test_a_type_the_matrix_failed_is_not_sent_back_to_capture():
    """A measured failure repeats if you re-run it; it needs replanning."""
    config = _asset_config([
        _episode("apartment_human_speaker", "room_apartment",
                 [("human", HUMAN), ("speaker", DEVICE)],
                 [{"qa_id": "QA-01", "target_instance_ids": ["source1"]}]),
    ])
    row = _fine_type_row("device/blender", ["ue_only_lamp"])
    row["matrix_status"] = "fail"
    row["blocking_dimensions"] = ["clearance", "placement"]
    plan = _plan(config, [row])
    assert plan["added_request_count"] == 0
    reason = plan["skipped"][0]["reason"]
    assert "needs replanning before any capture" in reason
    assert "clearance, placement" in reason


def test_a_type_the_owner_says_needs_no_new_capture_is_left_alone():
    """The planner cannot see retained evidence; the owner declares it."""
    config = _asset_config([
        _episode("apartment_human_dog", "room_apartment",
                 [("human", HUMAN), ("dog", ANIMAL)],
                 [{"qa_id": "QA-01", "target_instance_ids": ["source1"]}]),
    ])
    row = _fine_type_row("animal/dog/shiba_inu", ["cat"])
    inputs = {
        "config": config,
        "registry": _asset_registry(),
        "room_catalog": _asset_catalog(),
        "max_new_ordinary": 4,
        "max_new_core_groups": 0,
        "max_planned_requests": 4,
        "source_type_no_new_native": ["animal/dog/shiba_inu"],
    }
    plan = runner.plan_v1_next_requests(
        manifest={"episodes": [], "production": {}},
        feedback={"outstanding_requests": [row]},
        coverage_inputs=inputs,
    )
    assert plan["added_request_count"] == 0
    assert "needing no new native capture" in plan["skipped"][0]["reason"]

    # Without that declaration the same type is a perfectly good request.
    inputs.pop("source_type_no_new_native")
    again = runner.plan_v1_next_requests(
        manifest={"episodes": [], "production": {}},
        feedback={"outstanding_requests": [row]},
        coverage_inputs=inputs,
    )
    assert again["added_request_count"] == 1


# --------------------------------------------------------------------------
# C02-R2: registering a question is not the same as carrying it
#
# A template whose profile pins the anchor in view cannot host an
# out-of-view target. It still lists the qa_id, so picking on the listing
# alone attached the target to a request the sampler would refuse. The
# capability probe already knew that and was only ever recorded.
# --------------------------------------------------------------------------


def _visibility_episode(request_id, anchor_visibility, targets):
    episode = _episode(request_id, "room_apartment",
                       [("human", HUMAN), ("dog", ANIMAL)], targets)
    episode["profile"] = {"retry_budget_within_profile": 8,
                          "anchor_visibility": anchor_visibility}
    return episode


def test_a_template_whose_profile_cannot_carry_the_target_is_passed_over(
        monkeypatch):
    in_fov = _visibility_episode(
        "pinned_in_fov", "in_fov",
        [{"qa_id": "QA-08", "branch": "out_of_view",
          "target_instance_ids": ["source1"]}])
    off_screen = _visibility_episode(
        "allows_off_screen", "off_screen",
        [{"qa_id": "QA-08", "branch": "out_of_view",
          "target_instance_ids": ["source1"]}])
    config = _asset_config([in_fov, off_screen])

    def fake_probe(request, target, *, registry, inputs):
        visibility = (request.get("profile") or {}).get("anchor_visibility")
        if target is not None and str(target.get("branch")) == "out_of_view" \
                and visibility == "in_fov":
            return {"status": "blocked",
                    "candidate_states": [{"qa_id": "QA-08",
                                          "branch": "out_of_view",
                                          "state": NOT_APPLICABLE,
                                          "reason": "anchor is pinned in view"}]}
        return {"status": "pass", "capability_basis": {}}

    monkeypatch.setattr(runner, "_p19_planning_probe", fake_probe)
    plan = _plan(config, [
        {"kind": "qa_branch", "qa_id": "QA-08", "branch": "out_of_view",
         "state": "evidence_missing_or_unsampled"}])
    assert plan["added_request_count"] == 1, plan.get("skipped")
    entry = plan["requests"][0]
    assert entry["request"]["request_id"].startswith("allows_off_screen")
    # The template that could not carry it is named, not silently dropped.
    rejected = entry["planning"]["templates_rejected_before_this_one"]
    assert [row["request_id"] for row in rejected] == ["pinned_in_fov"]
    assert "cannot satisfy this target" in rejected[0]["reason"]


def test_when_no_template_can_carry_it_the_deficit_is_skipped_with_the_reasons(
        monkeypatch):
    config = _asset_config([
        _visibility_episode("pinned_in_fov", "in_fov",
                            [{"qa_id": "QA-08", "branch": "out_of_view",
                              "target_instance_ids": ["source1"]}]),
    ])
    monkeypatch.setattr(
        runner, "_p19_planning_probe",
        lambda request, target, *, registry, inputs: {
            "status": "blocked",
            "candidate_states": [{"qa_id": "QA-08", "state": NOT_APPLICABLE}]})
    plan = _plan(config, [
        {"kind": "qa_branch", "qa_id": "QA-08", "branch": "out_of_view",
         "state": "evidence_missing_or_unsampled"}])
    assert plan["added_request_count"] == 0
    skipped = plan["skipped"][0]
    assert "cannot satisfy this target" in skipped["reason"]
    assert skipped["templates_rejected"][0]["request_id"] == "pinned_in_fov"
    assert skipped["templates_rejected"][0]["candidate_states"]


def test_a_probe_that_passes_still_yields_the_first_registering_template(
        monkeypatch):
    """Nothing changes for a deficit whose obvious template is fine."""
    config = _asset_config([
        _episode("first", "room_apartment", [("human", HUMAN), ("dog", ANIMAL)],
                 [{"qa_id": "QA-01", "target_instance_ids": ["source1"]}]),
        _episode("second", "room_apartment", [("human", HUMAN), ("dog", ANIMAL)],
                 [{"qa_id": "QA-01", "target_instance_ids": ["source1"]}]),
    ])
    monkeypatch.setattr(
        runner, "_p19_planning_probe",
        lambda request, target, *, registry, inputs: {"status": "pass"})
    plan = _plan(config, [
        {"kind": "qa_id", "qa_id": "QA-01",
         "state": "evidence_missing_or_unsampled"}])
    assert plan["requests"][0]["request"]["request_id"].startswith("first")
    assert plan["requests"][0]["planning"][
        "templates_rejected_before_this_one"] == []

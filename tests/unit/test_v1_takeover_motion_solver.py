
"""T03a integration checks for the P03 motion solver in the P05 sampler."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pytest

from avengine.qa.answerability import MeshHandle
from avengine.rooms import conditioned_sampler as cs
from avengine.rooms.furniture_layout import clock_config
from avengine.rooms.walkable_space import RasterWalkableSpace
from avengine.routes.raster_pathfinder import RasterPathfinder

REPOSITORY = Path(__file__).resolve().parents[2]
M03_POOL_ROW = REPOSITORY / (
    "tmp/binding_v1_parallel_20260910/TAKEOVER/M03/"
    "attempt_20260910T181653Z_pid688995/real_readback_retry/"
    "alarm_beep_pool_row.json"
)


def _registry() -> dict:
    timeline = {
        "idle_action_id": "idle",
        "walking_action_id": "walk",
        "walk_phase_period_frames": 30,
        "body_plan_id": "test_body",
        "template_id": "test_template",
        "local_anatomical_forward_axis": [1.0, 0.0, 0.0],
    }
    return {
        "assets": [
            {
                "asset_id": "human_0",
                "revision": "v1",
                "entity_class": "articulated_human",
                "identity": {"species_id": "human"},
                "display_label": "human",
                "realized_attributes": {"sex_or_gender_label": "male"},
                "timeline": deepcopy(timeline),
                "default_emitter_anchor_id": "mouth",
                "emitter_anchors": [
                    {
                        "anchor_id": "mouth",
                        "offset_m": [0.0, 1.6, 0.0],
                        "offset_space": "final_scaled_asset_root",
                    }
                ],
            },
            {
                "asset_id": "dog_0",
                "revision": "v1",
                "entity_class": "articulated_animal",
                "identity": {"species_id": "dog"},
                "display_label": "dog",
                "realized_attributes": {"species_id": "dog"},
                "timeline": deepcopy(timeline),
                "default_emitter_anchor_id": "mouth",
                "emitter_anchors": [
                    {
                        "anchor_id": "mouth",
                        "offset_m": [0.0, 1.1, 0.0],
                        "offset_space": "final_scaled_asset_root",
                    }
                ],
            },
            {
                "asset_id": "desk_phone_0",
                "revision": "v1",
                "entity_class": "rigid_object",
                "identity": {"object_type": "desk_telephone", "category": "appliance"},
                "display_label": "desk telephone",
                "realized_attributes": {},
                "default_emitter_anchor_id": "body",
                "emitter_anchors": [
                    {
                        "anchor_id": "body",
                        "offset_m": [0.0, 0.75, 0.0],
                        "offset_space": "final_scaled_asset_root",
                    }
                ],
            },
        ]
    }


def _space() -> RasterWalkableSpace:
    pathfinder = RasterPathfinder(
        np.ones((40, 40), dtype=bool),
        bounds_m=[[0.0, -1.0, 0.0], [10.0, 1.0, 10.0]],
        floor_height_m=0.0,
    )
    return RasterWalkableSpace(
        pathfinder,
        {"floor_height_m": 0.0, "resolution_m": 0.25, "authority": "t03a_retained_grid"},
    )


def _request(qa_id: str, branch: str) -> dict:
    return {
        "episode_id": f"t03a_{qa_id}_{branch}",
        "seed": 19,
        "sampling_policy": cs.POLICY,
        "source_asset_ids": ["human_0", "dog_0", "desk_phone_0"],
        "entities": {
            "total_count": 3,
            "silent_count": 2,
            "min_articulated_count": 2,
            "instances": [
                {
                    "instance_id": "human_target",
                    "asset_id": "human_0",
                    "source_class": "articulated_human",
                    "role": "target",
                    "speaking": True,
                },
                {
                    "instance_id": "dog_competitor",
                    "asset_id": "dog_0",
                    "source_class": "articulated_animal",
                    "role": "competitor",
                    "speaking": False,
                },
                {
                    "instance_id": "device_competitor",
                    "asset_id": "desk_phone_0",
                    "source_class": "rigid_static_object",
                    "role": "competitor",
                    "speaking": False,
                },
            ],
        },
        "sound_selection": {
            "sound_class_config": {
                "human_nonverbal_sound_classes": ["alarm_beep"],
                "species_sound_classes": {"dog": ["alarm_beep"]},
                "object_sound_classes": {"desk_telephone": ["alarm_beep"]},
            }
        },
        "qa_ids": [qa_id],
        "question_branches": {qa_id: branch},
        "profile": {
            "separation_bin_deg": [15, 120],
            "reserve_tail_s": 1.0,
            "minimum_motion_s": 1.0,
            "retry_budget_within_profile": 120,
        },
        "public_time_precision": 0,
    }


def _plan(qa_id: str, branch: str) -> dict:
    if not M03_POOL_ROW.is_file():
        pytest.skip(f"M03 pool row is unavailable: {M03_POOL_ROW}")
    sound = json.loads(M03_POOL_ROW.read_text(encoding="utf-8"))
    sound["path"] = str((REPOSITORY / sound["path"]).resolve())
    return cs.build_conditioned_plan(
        room={"room_id": "t03a_fixture"},
        request=_request(qa_id, branch),
        source_registry=_registry(),
        sounds=[sound],
        space=_space(),
        mesh=MeshHandle(np.zeros((0, 3)), np.zeros((0, 3), dtype=int)),
        clock=clock_config(frame_count=150, frame_rate_hz=15, sample_rate_hz=16000),
    )


def _motion_states(plan: dict) -> dict[str, list[bool]]:
    states: dict[str, list[bool]] = {}
    for frame in plan["visual_plan"]["frames"]:
        for actor in frame["actor_states"]:
            states.setdefault(actor["entity_instance_id"], []).append(bool(actor["moving"]))
    return states


@pytest.mark.parametrize(
    ("qa_id", "branch", "expected_trend"),
    (
        ("QA-06", "moving", None),
        ("QA-15", "nearer", "negative"),
        ("QA-15", "farther", "positive"),
        ("QA-17", "yes", None),
        ("QA-17", "no", None),
    ),
)
def test_p03_solver_drives_p05_plan_for_human_animal_device(
    qa_id: str, branch: str, expected_trend: str | None,
) -> None:
    plan = _plan(qa_id, branch)
    solver = plan["planning_result"]["motion_solver"]
    assert solver["status"] == "solved"
    assert plan["question_condition_match"]["compiled"]
    assert "_compiled_conditions" not in plan["question_condition_match"]

    requirements = {
        row["entity_instance_id"]: row for row in solver["requirements"]
    }
    states = _motion_states(plan)
    for record in plan["activity_plan"]["actors"]:
        if record.get("motion") == "solver_conditioned_walk":
            assert record["motion_solver"]["speed_within_declared_range"] is True
            assert record["motion_solver"]["time_stretch_applied"] is False
    space = _space()
    for entity_id in requirements:
        points = [
            actor["root_transform"]["translation_m"]
            for frame in plan["visual_plan"]["frames"]
            for actor in frame["actor_states"]
            if actor["entity_instance_id"] == entity_id
        ]
        assert all(space.is_navigable(point) for point in points)
    target = requirements["human_target"]
    dog = requirements["dog_competitor"]
    device = requirements["device_competitor"]

    for entity_id, requirement in requirements.items():
        for first, last in requirement["still_frames"]:
            assert not any(states[entity_id][first:last])
        if requirement["moving_frames"] is not None:
            first, last = requirement["moving_frames"]
            assert all(states[entity_id][first:last])

    assert device["locomotion_state"] == "not_applicable_by_definition"
    assert device["moving_frames"] is None
    if qa_id == "QA-17" and branch == "no":
        assert device["semantics"] == "unconstrained"
        assert "cannot" in device["reason"]
    if qa_id == "QA-06":
        assert target["moving_frames"] is not None
        assert dog["moving_frames"] is None
    elif qa_id == "QA-15":
        assert target["moving_frames"] is not None
        trend = plan["planned_conditions"]["planned_distance_trend"]
        assert trend["sign"] == expected_trend
        assert trend["meets_margin"] is True
        assert trend["monotone_within_tolerance"] is True
        assert plan["planned_conditions"]["distance_trend_criterion_source"]
    elif branch == "yes":
        assert target["moving_frames"] is not None
        assert target["moving_frames"][0] == plan["planning_result"]["motion_solver"]["solutions"][0]["placements"][0]["end_frame"]
        assert dog["moving_frames"] is None
    else:
        assert target["moving_frames"] is None
        assert dog["moving_frames"] is not None

    sample_rate = plan["clock"]["sample_rate_hz"]
    for entity_id, start_s in solver["event_start_s"].items():
        event = next(
            row for row in plan["audio_events"]
            if row.get("entity_instance_id") == entity_id
        )
        assert start_s == pytest.approx(event["start_sample"] / sample_rate)

def test_solver_reuses_a_native_route_bank_polyline(tmp_path: Path) -> None:
    from avengine.rooms.walkable_space import NativeRouteWalkableSpace

    request = _request("QA-06", "moving")
    request["source_asset_ids"] = ["human_0", "human_0"]
    request["entities"]["total_count"] = 2
    request["entities"]["silent_count"] = 1
    request["entities"]["instances"] = [
        {
            "instance_id": "human_target",
            "asset_id": "human_0",
            "source_class": "articulated_human",
            "role": "target",
            "speaking": True,
        },
        {
            "instance_id": "human_competitor",
            "asset_id": "human_0",
            "source_class": "articulated_human",
            "role": "competitor",
            "speaking": False,
        },
    ]
    sound = json.loads(M03_POOL_ROW.read_text(encoding="utf-8"))
    sound["path"] = str((REPOSITORY / sound["path"]).resolve())
    routes = [
        {
            "route_id": "native_a",
            "points_m": [[1.0 + 0.55 * value, 0.0, 1.0] for value in range(7)],
        },
        {
            "route_id": "native_b",
            "points_m": [[1.0 + 0.55 * value, 0.0, 2.0] for value in range(7)],
        },
    ]
    base = _space()
    native = NativeRouteWalkableSpace(
        base.pathfinder, base.metadata, routes, frame_rate_hz=15.0,
    )
    plan = cs.build_conditioned_plan(
        room={"room_id": "t03a_native_fixture"},
        request=request,
        source_registry=_registry(),
        sounds=[sound],
        space=native,
        mesh=MeshHandle(np.zeros((0, 3)), np.zeros((0, 3), dtype=int)),
        clock=clock_config(frame_count=150, frame_rate_hz=15, sample_rate_hz=16000),
    )
    assert plan["activity_plan"]["selected_route_points_m"]
    assert any(
        record.get("motion") == "solver_conditioned_walk"
        and record["motion_solver"]["speed_within_declared_range"] is True
        for record in plan["activity_plan"]["actors"].values()
    )



def test_qa17_no_keeps_the_device_in_a_human_animal_device_world():
    """A device that cannot walk must not void the animal standing next to it.

    unified_catalog._p8_apply_distractor_gate defers a candidate only when
    every real distractor shares the gold answer, so this branch needs one
    competitor that moved, not all of them. The animal carries that duty and the
    desk telephone is left unconstrained with the reason recorded on it.
    """
    plan = _plan("QA-17", "no")
    solver = plan["planning_result"]["motion_solver"]
    assert solver["status"] == "solved"

    duty = {row["entity_instance_id"]: row for row in solver["requirements"]}
    assert set(duty) == {"human_target", "dog_competitor", "device_competitor"}

    # The target answers "no", so it stays still over the interval the question reads.
    assert duty["human_target"]["must_move"] is False

    # One capable competitor carries the differing answer.
    assert duty["dog_competitor"]["must_move"] is True

    # The device is neither commanded to walk nor allowed to void the group.
    assert duty["device_competitor"]["must_move"] is False
    assert duty["device_competitor"]["semantics"] == "unconstrained"
    assert "another competitor" in duty["device_competitor"]["reason"].lower()

    states = _motion_states(plan)
    assert any(states["dog_competitor"]), "the competitor holding the duty never moved"
    assert not any(states["device_competitor"]), "a registered static object walked"
    assert not any(states["human_target"]), "the no branch requires a still target"


def test_a_branch_that_no_competitor_can_answer_differently_is_refused():
    """The refusal fires when nothing in the room can move, not when one thing cannot."""
    request = _request("QA-17", "no")
    request["entities"]["instances"] = [
        row for row in request["entities"]["instances"]
        if row["instance_id"] != "dog_competitor"
    ]
    request["entities"]["total_count"] = len(request["entities"]["instances"])
    keep = {row["asset_id"] for row in request["entities"]["instances"]}
    if request.get("source_asset_ids") is not None:
        request["source_asset_ids"] = [a for a in request["source_asset_ids"] if a in keep]
    request["entities"].pop("silent_count", None)
    request["entities"].pop("min_articulated_count", None)
    request.pop("silent_actor_count", None)
    with pytest.raises(cs.ConditionedRequestConflict) as raised:
        cs.resolve_conditioned_request(request, _registry(), generator=cs)
    assert {"competitor_motion"} == {row["knob"] for row in raised.value.conflicts}
    assert "registered static objects" in str(raised.value)

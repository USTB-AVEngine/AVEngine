"""L37 constructive motion checks for both QA-06 branches."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np


REPOSITORY = Path(__file__).resolve().parents[2]


def _load_fixture(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(
        name, REPOSITORY / "tests/unit" / filename
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _build(branch: str, *, constructive: bool):
    fixture = _load_fixture(
        "l37_takeover_motion_solver", "test_v1_takeover_motion_solver.py"
    )
    from avengine.qa.answerability import MeshHandle
    from avengine.rooms.furniture_layout import clock_config

    request = fixture._request("QA-06", branch)
    request["profile"]["constructive_motion"] = constructive
    sound = json.loads(fixture.M03_POOL_ROW.read_text(encoding="utf-8"))
    sound["path"] = str((fixture.REPOSITORY / sound["path"]).resolve())
    return fixture.cs.build_conditioned_plan(
        room={"room_id": "l37_fixture"},
        request=request,
        source_registry=fixture._registry(),
        sounds=[sound],
        space=fixture._space(),
        mesh=MeshHandle(np.zeros((0, 3)), np.zeros((0, 3), dtype=int)),
        clock=clock_config(frame_count=150, frame_rate_hz=15, sample_rate_hz=16000),
    )


def _motion_runs(plan: dict) -> dict[str, list[tuple[int, int]]]:
    flags: dict[str, list[bool]] = {}
    for frame in plan["visual_plan"]["frames"]:
        for actor in frame["actor_states"]:
            flags.setdefault(actor["entity_instance_id"], []).append(
                bool(actor["moving"])
            )
    result = {}
    for entity_id, values in flags.items():
        runs = []
        start = None
        for index, value in enumerate(values + [False]):
            if value and start is None:
                start = index
            elif not value and start is not None:
                runs.append((start, index))
                start = None
        result[entity_id] = runs
    return result


def _requirements(plan: dict) -> dict[str, dict]:
    return {
        row["entity_instance_id"]: row
        for row in plan["planning_result"]["motion_solver"]["requirements"]
    }


def test_still_branch_constructs_one_target_window_and_one_moving_foil():
    plan = _build("still", constructive=True)
    requirements = _requirements(plan)
    runs = _motion_runs(plan)

    assert plan["condition_profile"]["constructive_motion"] is True
    assert plan["activity_plan"]["motion_construction"] == (
        "constructive_existing_navigation"
    )
    assert len(plan["audio_events"]) == 1
    assert plan["audio_events"][0]["entity_instance_id"] == "human_target"
    target_window = tuple(requirements["human_target"]["still_frames"][0])
    foil_window = tuple(requirements["dog_competitor"]["moving_frames"])
    assert target_window == foil_window
    assert runs["human_target"] == []
    assert runs["dog_competitor"] == [foil_window]
    assert runs["device_competitor"] == []
    for actor in plan["activity_plan"]["actors"]:
        if actor.get("motion") == "solver_conditioned_walk":
            assert actor["all_sampled_centers_navigable"] is True
            assert actor["motion_solver"]["window_reads_moving"] is True


def test_moving_branch_constructs_the_target_window_and_keeps_foil_still():
    plan = _build("moving", constructive=True)
    requirements = _requirements(plan)
    runs = _motion_runs(plan)

    assert plan["condition_profile"]["constructive_motion"] is True
    assert len(plan["audio_events"]) == 1
    target_window = tuple(requirements["human_target"]["moving_frames"])
    assert target_window == tuple(requirements["dog_competitor"]["still_frames"][0])
    assert runs["human_target"] == [target_window]
    assert runs["dog_competitor"] == []
    assert runs["device_competitor"] == []


def test_constructive_motion_is_opt_in_and_default_route_consumption_survives():
    legacy = _build("still", constructive=False)
    assert legacy["condition_profile"]["constructive_motion"] is False
    assert any(
        actor.get("motion_construction") == "legacy_random_route_then_solver"
        for actor in legacy["activity_plan"]["actors"]
        if actor.get("motion") == "solver_conditioned_walk"
    )


def test_constructive_motion_is_deterministic_for_a_fixed_input():
    first = _build("still", constructive=True)
    second = _build("still", constructive=True)

    def relevant(plan):
        return {
            "events": [
                {
                    key: row[key]
                    for key in ("entity_instance_id", "start_sample", "end_sample")
                }
                for row in plan["audio_events"]
            ],
            "actors": [
                {
                    key: row[key]
                    for key in ("entity_instance_id", "root_transform", "moving")
                }
                for frame in plan["visual_plan"]["frames"]
                for row in frame["actor_states"]
            ],
        }

    assert relevant(first) == relevant(second)


def test_compiled_still_duty_selects_one_capable_competitor():
    fixture = _load_fixture(
        "l37_conditioned_motion", "test_v1_conditioned_motion.py"
    )
    three_humans = fixture.TWO_HUMANS + [
        {
            "entity_instance_id": "inst_3",
            "asset_id": "human_a",
            "source_class": "articulated_human",
        }
    ]
    solution = fixture._solve("QA-06", "still", instances=three_humans)
    requirements = {
        row.entity_instance_id: row for row in solution.requirements
    }

    assert requirements["inst_1"].must_move is False
    assert requirements["inst_2"].must_move is True
    assert requirements["inst_3"].must_move is False
    assert requirements["inst_3"].semantics == "unconstrained"

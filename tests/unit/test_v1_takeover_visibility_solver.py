"""T03b takeover checks for the real P04-to-sampler visibility path."""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pytest

from avengine.qa import generation_conditions as gc
from avengine.rooms import conditioned_sampler as cs
from avengine.rooms import conditioned_visibility as cv
from avengine.rooms.qa_episode import compile_question_conditions
from avengine.rooms.furniture_layout import clock_config
from avengine.capture.qa_plan_adapters import load_planning_resources_for_room


ROOT = Path(__file__).resolve().parents[2]
REAL_BASE_PLAN = (
    ROOT
    / "tmp/binding_v1_parallel_20260910/TAKEOVER/T03a/"
      "attempt_20260910T185938Z_pid734148/real_registered_retry/"
      "real_registered_plan.json"
)
REAL_REGISTRY = ROOT / "examples/runtime/source_asset_runtime_profiles.json"
REAL_SOUND_POOL = Path(
    "/data/datasets/avengine_workspaces/multi_home_activity_20260905/root/"
    "qa_pilot46_10s_peak_dbfs3_20260908_batch_prepare_v1/batch_sounds.json"
)


def _camera(candidate_id: str) -> cv.CameraPose:
    return cv.CameraPose(
        candidate_id=candidate_id,
        position_m=(0.0, 1.55, 0.0),
        forward=(0.0, 0.0, -1.0),
        right=(1.0, 0.0, 0.0),
        up=(0.0, 1.0, 0.0),
        horizontal_fov_deg=85.0,
        resolution_hw=(720, 1280),
    )


def test_max_candidates_also_bounds_expensive_ray_poses() -> None:
    track = cv.ActorTrack(
        instance_id="source1",
        positions_m=np.repeat([[0.0, 0.0, -4.0]], 8, axis=0),
        body=cv.BodyProxy(height_m=1.6),
    )
    requirement = cv.VisibilityRequirement(
        kind="visibility_state",
        subject="source1",
        state="visible_clear",
    )
    solved = cv.solve_visibility_candidates(
        [requirement],
        camera_poses=[_camera("grid_00000_yaw_000"), _camera("grid_00001_yaw_000")],
        tracks=[track],
        frame_rate_hz=15.0,
        max_candidates=1,
        max_ray_poses=8,
    )
    assert solved["budgets"]["expensive_pose_budget"] == 1
    assert solved["stages"]["poses_screened_with_rays"] <= 1
    assert len(solved["candidates"]) <= 1


@pytest.fixture(scope="module")
def real_visibility_context() -> dict:
    assert REAL_BASE_PLAN.is_file(), REAL_BASE_PLAN
    assert REAL_REGISTRY.is_file(), REAL_REGISTRY
    assert REAL_SOUND_POOL.is_file(), REAL_SOUND_POOL
    base = json.loads(REAL_BASE_PLAN.read_text(encoding="utf-8"))
    request = deepcopy(base["request"])
    request.update(
        {
            "episode_id": "t03b_real_hm3d_qa09_no",
            "qa_ids": ["QA-09"],
            "question_branches": {"QA-09": "no"},
        }
    )
    request["profile"] = {
        **request.get("profile", {}),
        "anchor_line_of_sight": "occluded",
        "speech_motion": "speaker_moving",
        "competitor_motion": "still",
        "retry_budget_within_profile": 8,
    }
    request["camera"] = {
        **request.get("camera", {}),
        "visibility_solver": {
            "max_candidates": 8,
            "max_ray_poses": 16,
            "candidate_pool_budget": 512,
            "frame_budget": 150,
        },
    }
    registry = json.loads(REAL_REGISTRY.read_text(encoding="utf-8"))
    sounds = json.loads(REAL_SOUND_POOL.read_text(encoding="utf-8"))["sounds"]
    space, mesh, layout, resolution = load_planning_resources_for_room(
        request["room_id"], request
    )
    clock = clock_config(
        frame_count=int(request["frame_count"]),
        frame_rate_hz=float(request["frame_rate_hz"]),
        sample_rate_hz=int(request["sample_rate_hz"]),
    )
    plan = cs.build_conditioned_plan(
        room={
            **layout,
            "room_package": resolution.planning_room.get("room_package"),
        },
        request=request,
        source_registry=registry,
        sounds=sounds,
        space=space,
        mesh=mesh,
        clock=clock,
    )
    return {
        "base": base,
        "request": request,
        "registry": registry,
        "sounds": sounds,
        "space": space,
        "mesh": mesh,
        "room": {
            **layout,
            "room_package": resolution.planning_room.get("room_package"),
        },
        "clock": clock,
        "plan": plan,
    }


def test_real_registered_hm3d_qa09_routes_camera_through_p04(
    real_visibility_context: dict,
) -> None:
    plan = real_visibility_context["plan"]
    report = plan["planned_conditions"]["visibility_solver"]
    assert report["status"] == "screened_no_pixels_rendered"
    assert report["geometry_authority"] == "acoustic_proxy_mesh"
    assert [
        row["kind"] for row in report["requirements"]
    ] == ["fully_occluded_without_return"]
    # C07: the row's ``state`` is the pixel vocabulary and ``capability_state``
    # is the planner's. They used to share one key, so a requirement that asked
    # for a named pixel state read back as the capability word "available" and
    # a reader could not tell which state the plan had actually solved.
    row = report["requirements"][0]
    assert row["state"] is None, "this kind carries no explicit pixel state"
    assert row["capability_state"] == "available"
    assert row["required_dynamics"] == []
    assert report["budgets"]["frame_evaluations_per_ray_pose"] == 150
    assert report["budgets"]["frame_budget"] == 150
    assert (
        report["stages"]["poses_screened_with_rays"]
        <= report["budgets"]["expensive_pose_budget"]
    )
    bounded = report["bounded_selection"]
    assert bounded["mode"] == "bounded_projection_then_ray"
    assert bounded["candidate_pool_count"] <= bounded["candidate_pool_budget"]
    assert bounded["ray_pose_count"] <= 16
    assert set(bounded["ray_pose_ids"]) <= set(bounded["candidate_pool_ids"])
    sampler_stages = plan["planned_conditions"]["stages"]
    assert sampler_stages["cheap_candidate_pool_count"] == bounded["candidate_pool_count"]
    assert sampler_stages["bounded_ray_candidate_count"] == bounded["ray_pose_count"]
    assert sampler_stages["ray_poses_evaluated"] == bounded["ray_pose_count"]
    assert bounded["ray_frame_evaluations"] == sampler_stages["ray_frame_evaluations"]
    assert bounded["ray_los_queries"] == sampler_stages["ray_los_queries"]
    assert report["candidate_ids_applied"]
    selected = plan["visual_plan"]["camera"]["candidate_id"]
    assert selected in report["candidate_ids_applied"]
    assert all(
        candidate["verdict"] != "refuted"
        for candidate in report["candidates"]
    )
    assert report["native_acceptance"]["status"] == "not_run"
    assert (
        report["native_acceptance"]["entrypoint"]
        == "avengine.rooms.conditioned_visibility.accept_native_visibility"
    )
    assert plan["evidence_status"] == {
        "native_visual": "not_run",
        "native_audio": "not_run",
        "qa_validity": "not_run",
    }


def test_real_compiled_qa07_keeps_entry_side_and_integer_window_requirement(
    real_visibility_context: dict,
) -> None:
    context = real_visibility_context
    plan = context["plan"]
    instances = plan["entity_instances"]
    compiled = compile_question_conditions(
        ["QA-07"],
        instances,
        branches={"QA-07": "right"},
        registry=context["registry"],
        generator=cs,
        public_time_precision=0,
        include_compiled=True,
    )["_compiled_conditions"]
    requirements = cv.requirements_from_conditions(
        compiled[0].to_dict(),
        observation_windows_by_subject={
            str(item.entity_instance_id): [[0, int(context["clock"]["frame_count"])]]
            for item in compiled[0].subjects
        },
        public_time_precision=0,
    )
    entry = next(
        item for item in requirements if item.kind == "out_of_view_to_visible"
    )
    assert entry.subject == "human_target"
    assert entry.side == "right"
    assert entry.require_publishable_window is True
    assert entry.observation_windows == ((0, 150),)
    assert entry.as_report()["confirmation_authority"] == "native_pixel_witness"


def test_a_named_pixel_state_survives_the_plan_readback() -> None:
    """A ``fully_occluded`` request must read back as ``fully_occluded``.

    The same shape C01-R2 could not finish chasing: the requirement that
    reaches the solver is a real pixel state, and the plan's serialized row
    has to say so rather than reporting the planner's capability verdict in
    its place.
    """

    requirement = cv.VisibilityRequirement(
        kind="visibility_state",
        subject="human_target",
        state="fully_occluded",
        qa_id="QA-24",
    )
    track = cv.ActorTrack(
        instance_id="human_target",
        positions_m=np.repeat([[0.0, 0.0, -4.0]], 8, axis=0),
        body=cv.BodyProxy(height_m=1.6),
    )
    solved = cv.solve_visibility_candidates(
        [requirement],
        camera_poses=[_camera("grid_00000_yaw_000")],
        tracks=[track],
        frame_rate_hz=15.0,
    )
    serialized = json.loads(json.dumps(solved))
    row = serialized["requirements"][0]
    assert row["state"] == "fully_occluded"
    assert row["capability_state"] == "available"
    with pytest.raises(cv.ConditionedVisibilityError):
        cv.VisibilityRequirement(
            kind="visibility_state", subject="human_target", state="available"
        )

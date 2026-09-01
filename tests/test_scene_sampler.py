"""Scene-agnostic sampler: solved answer bands, and every gate proven to bite.

用两个**合成场景**(不引用任何真实房间)证明:同一套代码在不同场景直接
运行;答案带是解出来的而非撞出来的(每个带都能构造,包括固定机位下曾经
零可行的那一侧);每条约束都有阳性对照;拒绝有明确原因。
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools" / "qa"))

from scene_sampler import (  # noqa: E402
    FRAME_COUNT,
    RejectionLedger,
    Rejection,
    Route,
    SceneInputs,
    circular_gap_deg,
    interior_answer_band,
    line_of_sight_from_feasible_grid,
    load_scene,
    open_angle_gold_regions_disjoint,
    relative_azimuth_deg,
    solve_backward_cross_time,
    solve_forward_cross_time,
    solve_instant_binding,
    solve_distance_change_pair,
    solve_instant_distance_order,
    solve_motion_state_pair,
    solve_instant_azimuth,
    yaw_interval_for_band,
    _pick_other_route,
)

PARAMS = {"THETA_FULL": 15.0, "THETA_HALF": 30.0,
          "MIN_AZIMUTH_SEP": 25.0,
          "MIN_CAMERA_DISTANCE_CM": 100.0}
BANDS = [(-52.5, -17.5), (-17.5, 17.5), (17.5, 52.5)]


def straight_route(route_id, start, end, speed=0.7):
    samples = []
    for i in range(FRAME_COUNT):
        t = i / (FRAME_COUNT - 1)
        samples.append((start[0] + (end[0] - start[0]) * t,
                        start[1] + (end[1] - start[1]) * t))
    return Route(route_id, samples, speed)


def synthetic_scene(scene_id="synth_a", spread=600.0, n=12, hfov=105.0):
    """一圈可站点 + 一束穿过中心的直线路线;与任何真实房间无关。"""
    count = max(n, 24)
    points = [(spread * math.cos(2 * math.pi * k / count),
               spread * math.sin(2 * math.pi * k / count))
              for k in range(count)]
    routes = []
    for k in range(count):
        a = points[k]
        # Neighbouring paths can remain jointly visible.  Diameter-only paths
        # force every pair to opposite sides and are not a valid dual-motion
        # positive fixture.
        b = points[(k + 4) % count]
        routes.append(straight_route(f"{scene_id}_r{k}", a, b))
    return SceneInputs(scene_id=scene_id, backend="synthetic", routes=routes,
                       stand_points=points, camera_points=points,
                       camera_height_m=1.47, hfov_deg=hfov)


def test_yaw_interval_solves_the_band_exactly():
    camera, point = (0.0, 0.0), (100.0, 0.0)
    lo, hi = yaw_interval_for_band(camera, point, 17.5, 52.5)
    for frac in (0.01, 0.5, 0.99):
        yaw = lo + (hi - lo) * frac
        az = relative_azimuth_deg(camera, yaw, point)
        assert 17.5 <= az < 52.5, (frac, az)


@pytest.mark.parametrize("band", BANDS)
def test_every_declared_band_is_constructible(band):
    """固定机位下右侧带曾经零可行 —— 解 yaw 之后每个带都能造出来。"""
    scene = synthetic_scene()
    rng = np.random.default_rng(7)
    ledger = RejectionLedger()
    plan = solve_forward_cross_time(scene, PARAMS, answer_band=band,
                                    answer_bands=BANDS,
                                    anchor_frame=45, idle_choices=(0, 8, 16),
                                    rng=rng, ledger=ledger)
    assert not isinstance(plan, Rejection), (band, ledger.summary())
    lo, hi = band
    assert lo <= plan.answer_cell["value_deg"] < hi
    assert plan.checks["azimuth_travel_deg"] > PARAMS["THETA_FULL"]
    assert plan.checks["anchor_separation_deg"] >= PARAMS["MIN_AZIMUTH_SEP"]
    assert plan.checks["gatea_open_gold_separation_deg"] > \
        2 * PARAMS["THETA_HALF"]
    assert plan.target_route.displacement_cm > 0
    assert plan.other_route.displacement_cm > 0
    assert plan.target_route.route_id != plan.other_route.route_id


def test_open_gold_separation_is_strict_at_double_half_width():
    """A changed number is not enough when the two credit bands overlap."""
    assert not open_angle_gold_regions_disjoint(0.0, 59.999, 30.0)
    assert not open_angle_gold_regions_disjoint(0.0, 60.0, 30.0)
    assert open_angle_gold_regions_disjoint(0.0, 60.001, 30.0)
    assert open_angle_gold_regions_disjoint(179.0, -119.0, 30.0)
    # A linear abs(a-b) implementation returns 340 and would wrongly pass.
    assert not open_angle_gold_regions_disjoint(170.0, -170.0, 30.0)


def test_gatea_actor_outside_declared_mcq_space_is_rejected():
    scene = synthetic_scene(hfov=180.0)
    def polar(degrees, radius):
        angle = math.radians(degrees)
        return (radius * math.cos(angle), radius * math.sin(angle))
    target = straight_route("target", polar(-40.0, 300.0),
                            polar(-40.0, 350.0))
    candidate = straight_route("candidate", polar(25.0, 300.0),
                               polar(25.0, 350.0))
    scene.routes = [candidate]  # +25deg is in the deliberate MCQ band gap
    ledger = RejectionLedger()
    result = _pick_other_route(
        scene, target, (0.0, 0.0), 0.0, -40.0, -40.0, -50.0, -30.0,
        [(-50.0, -30.0), (30.0, 50.0)], 25.0, 90.0, 30.0,
        PARAMS, 0, 74, np.random.default_rng(1), ledger)
    assert result is None
    assert "no_second_actor_in_declared_mcq_space" in \
        ledger.summary()["by_reason"]


def test_same_code_runs_on_a_second_scene_without_changes():
    """场景无关的最小证据:换一个几何完全不同的场景,同一调用直接跑。"""
    for scene in (synthetic_scene("synth_a", spread=600.0, n=12),
                  synthetic_scene("synth_b", spread=250.0, n=8)):
        rng = np.random.default_rng(7)
        ledger = RejectionLedger()
        plan = solve_forward_cross_time(scene, PARAMS, answer_band=BANDS[1],
                                        answer_bands=BANDS,
                                        anchor_frame=45, idle_choices=(0, 8, 16),
                                        rng=rng, ledger=ledger)
        assert not isinstance(plan, Rejection), (scene.scene_id,
                                                 ledger.summary())
        assert plan.scene_id == scene.scene_id


def test_narrow_fov_rejects_outer_band_with_named_reason():
    """视野窄到装不下外侧带 → 必须给出"带在视锥外"这条拒绝原因。"""
    scene = synthetic_scene(hfov=40.0)          # 半视锥 20°
    ledger = RejectionLedger()
    plan = solve_forward_cross_time(scene, PARAMS, answer_band=(17.5, 52.5),
                                    answer_bands=BANDS,
                                    anchor_frame=45, idle_choices=(0,),
                                    rng=np.random.default_rng(1),
                                    ledger=ledger, max_attempts=200)
    assert isinstance(plan, Rejection)
    assert "answer_band_outside_fov" in ledger.summary()["by_reason"]


def test_visual_fov_margin_reserves_space_for_actor_extent():
    scene = synthetic_scene(hfov=105.0)
    params = {**PARAMS, "VISUAL_FOV_MARGIN_DEG": 35.0}
    ledger = RejectionLedger()
    plan = solve_forward_cross_time(
        scene, params, answer_band=(17.5, 52.5), answer_bands=BANDS,
        anchor_frame=45, idle_choices=(0,), rng=np.random.default_rng(1),
        ledger=ledger, max_attempts=100,
    )
    assert isinstance(plan, Rejection)
    assert "answer_band_outside_fov" in ledger.summary()["by_reason"]


def test_answer_band_interior_margin_keeps_numeric_truth_off_boundaries():
    assert interior_answer_band(-17.5, 17.5, {
        "ANSWER_BAND_INTERIOR_MARGIN_DEG": 0.25
    }) == (-17.25, 17.25)
    with pytest.raises(ValueError, match="does not fit"):
        interior_answer_band(-1.0, 1.0, {
            "ANSWER_BAND_INTERIOR_MARGIN_DEG": 1.0
        })


def test_static_target_rejected_for_insufficient_travel():
    """目标不动 → 锚时即可读答案,错时题不成立。"""
    scene = synthetic_scene()
    scene.routes = [Route("static", [(300.0, 40.0)] * FRAME_COUNT, 0.0)]
    ledger = RejectionLedger()
    plan = solve_forward_cross_time(scene, PARAMS, answer_band=BANDS[1],
                                    answer_bands=BANDS,
                                    anchor_frame=45, idle_choices=(0,),
                                    rng=np.random.default_rng(2),
                                    ledger=ledger, max_attempts=200)
    assert isinstance(plan, Rejection)
    assert "target_route_static_for_dual_motion" in \
        ledger.summary()["by_reason"]


def test_single_stand_point_rejects_for_no_separable_second_actor():
    scene = synthetic_scene()
    # 只留一个可站点,且把它放在与路线中点重合的位置 → 分离不达标
    scene.stand_points = [scene.routes[0].at(45)]
    ledger = RejectionLedger()
    plan = solve_forward_cross_time(scene, PARAMS, answer_band=BANDS[1],
                                    answer_bands=BANDS,
                                    anchor_frame=45, idle_choices=(0,),
                                    rng=np.random.default_rng(5),
                                    ledger=ledger, max_attempts=200)
    assert isinstance(plan, Rejection)
    assert "no_separable_second_actor" in ledger.summary()["by_reason"]


def test_occlusion_screen_is_used_when_provided_and_reported_when_not():
    scene = synthetic_scene()
    ledger = RejectionLedger()
    plan = solve_forward_cross_time(scene, PARAMS, answer_band=BANDS[1],
                                    answer_bands=BANDS,
                                    anchor_frame=45, idle_choices=(0, 8, 16),
                                    rng=np.random.default_rng(7),
                                    ledger=ledger)
    assert not isinstance(plan, Rejection)
    assert plan.checks["line_of_sight_screened"] is False   # 未筛须如实标注

    blind = synthetic_scene()
    blind.line_of_sight = lambda a, b: False                # 全遮挡
    ledger2 = RejectionLedger()
    plan2 = solve_forward_cross_time(blind, PARAMS, answer_band=BANDS[1],
                                     answer_bands=BANDS,
                                     anchor_frame=45, idle_choices=(0, 8, 16),
                                     rng=np.random.default_rng(7),
                                     ledger=ledger2, max_attempts=300)
    assert isinstance(plan2, Rejection)
    assert "target_occluded_at_anchor_frame" in ledger2.summary()["by_reason"]


def test_backward_cross_time_queries_an_earlier_frame():
    scene = synthetic_scene()
    ledger = RejectionLedger()
    plan = solve_backward_cross_time(scene, PARAMS, answer_band=BANDS[0],
                                     answer_bands=BANDS,
                                     anchor_frame=68, query_frame=30,
                                     idle_choices=(0, 8),
                                     rng=np.random.default_rng(13),
                                     ledger=ledger)
    assert not isinstance(plan, Rejection), ledger.summary()
    assert plan.query_frame < plan.anchor_frame        # 视觉查询在音频锚之前
    assert plan.profile_id == "card1B"
    assert plan.checks["requires_silence_near_query"] is True
    lo, hi = BANDS[0]
    assert lo <= plan.answer_cell["value_deg"] < hi
    assert plan.other_route.displacement_cm > 0


def test_instant_binding_uses_two_moving_routes():

    scene = synthetic_scene()
    ledger = RejectionLedger()
    plan = solve_instant_binding(
        scene, PARAMS, instants=[12, 40], profile_id="card9",
        idle_choices=(0, 8), rng=np.random.default_rng(22), ledger=ledger)
    assert not isinstance(plan, Rejection), ledger.summary()
    assert plan.target_route.route_id != plan.other_route.route_id
    assert plan.other_route.displacement_cm > 0
@pytest.mark.parametrize("band", (BANDS[0], BANDS[2]))
def test_instant_azimuth_solves_declared_side_band_and_gatea_open(band):
    scene = synthetic_scene()
    ledger = RejectionLedger()
    side_bands = (BANDS[0], BANDS[2])
    plan = solve_instant_azimuth(
        scene, PARAMS, answer_band=band, answer_bands=side_bands,
        query_frame=30, profile_id="card2", idle_choices=(0, 8),
        rng=np.random.default_rng(33), ledger=ledger)
    assert not isinstance(plan, Rejection), ledger.summary()
    lo, hi = band
    assert lo <= plan.answer_cell["value_deg"] < hi
    assert plan.query_frame == plan.anchor_frame == 30
    assert plan.checks["gatea_open_gold_separation_deg"] > (
        2 * PARAMS["THETA_HALF"])
    assert plan.target_route.route_id != plan.other_route.route_id




def test_instant_binding_rejects_static_only_route_bank_for_dual_motion():
    scene = synthetic_scene()
    scene.routes = [
        Route(f"static_{index}", [(200.0 + index, 10.0)] * FRAME_COUNT, 0.0)
        for index in range(4)
    ]
    ledger = RejectionLedger()
    plan = solve_instant_binding(
        scene, PARAMS, instants=[12, 40], profile_id="card9",
        idle_choices=(0,), rng=np.random.default_rng(4), ledger=ledger,
        max_attempts=20,
    )
    assert isinstance(plan, Rejection)
    assert ledger.summary()["by_reason"] == {
        "target_route_static_for_dual_motion": 20
    }


def test_idle_shift_preserves_speed_and_endpoint_order():
    route = straight_route("r", (0.0, 0.0), (300.0, 0.0))
    shifted = route.shifted(10)
    assert shifted.at(0) == route.at(0)
    assert shifted.at(9) == route.at(0)          # 前 10 帧静止
    assert shifted.at(10) == route.at(0)
    step_before = circular_gap_deg(0, 0)          # 占位,证明可调用
    assert step_before == 0
    d_orig = math.dist(route.at(20), route.at(21))
    d_shift = math.dist(shifted.at(30), shifted.at(31))
    assert abs(d_orig - d_shift) < 1e-9          # 速度不变


def test_rejection_ledger_reports_reasons_and_examples():
    ledger = RejectionLedger()
    ledger.add(Rejection("a", "first a"))
    ledger.add(Rejection("a", "second a"))
    ledger.add(Rejection("b"))
    summary = ledger.summary()
    assert summary["total"] == 3
    assert list(summary["by_reason"]) == ["a", "b"]
    assert summary["first_example"]["a"] == "first a"


def test_load_scene_refuses_incomplete_config(tmp_path):
    with pytest.raises(ValueError) as exc:
        load_scene({"scene_id": "x", "backend": "y"})
    assert "missing keys" in str(exc.value)

    req = tmp_path / "req.json"
    req.write_text(json.dumps({}))

    # 未知库 schema:必须点名"按 schema 加适配器,而不是按房间 ID"
    unknown = tmp_path / "unknown.json"
    unknown.write_text(json.dumps({"schema": "t", "routes": []}))
    with pytest.raises(ValueError) as exc2:
        load_scene({"scene_id": "x", "backend": "y",
                    "route_bank": str(unknown), "camera_base_request": str(req)})
    assert "no route-bank adapter" in str(exc2.value)
    assert "room id" in str(exc2.value)

    # 已知 schema 但帧数不对:不能悄悄混入
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"schema": "avengine_apartment_route_bank_v1",
                               "routes": [{"route_id": "r1",
                                           "samples_ue_cm": [[0, 0]] * 10}]}))
    with pytest.raises(ValueError) as exc3:
        load_scene({"scene_id": "x", "backend": "y", "route_bank": str(bad),
                    "camera_base_request": str(req)})
    assert "no usable" in str(exc3.value)


def test_habitat_bank_adapter_normalises_metres_and_plane(tmp_path):
    """habitat 库:米→厘米、取 (x,z) 水平面;同一套 Route 出来。"""
    from scene_sampler import routes_from_bank
    path = [[0.0, 1.0, 0.0]] * 37 + [[2.0, 1.0, -3.0]] * 38
    bank = {"schema": "avengine_room_trajectory_bank_v2",
            "episodes": [{"episode_id": "e1",
                          "source_center_paths_m": {"source1": path}}]}
    routes = routes_from_bank(bank)
    assert len(routes) == 1
    assert routes[0].route_id == "e1:source1"
    assert routes[0].at(0) == (0.0, 0.0)
    assert routes[0].at(74) == (200.0, -300.0)      # 米→厘米,(x,z) 平面


def test_feasible_grid_los_blocks_obstacles_without_room_specific_code(tmp_path):
    metadata = tmp_path / "feasible.json"
    metadata.write_text(json.dumps({
        "source1": {
            "schema": "avengine_room_feasible_region_v1",
            "mask_shape_hw": [5, 6],
            "bounds_m": [[-0.05, 0.0, -0.05], [0.25, 2.0, 0.20]],
            "pixel_size_x_m": 0.05,
            "pixel_size_z_m": 0.05,
        }
    }))
    mask = np.ones((5, 6), dtype=np.uint8)
    mask[2, 3] = 0
    arrays = tmp_path / "feasible.npz"
    np.savez_compressed(arrays, feasible_mask=mask)
    config = {
        "metadata": str(metadata),
        "metadata_key": "source1",
        "arrays": str(arrays),
        "mask_key": "feasible_mask",
        "coordinate_contract": "habitat_xz_m_to_ue_xy_cm_v1",
    }
    los = line_of_sight_from_feasible_grid(config)
    assert los((0.0, 0.0), (20.0, 0.0)) is True
    assert los((0.0, 0.0), (20.0, 20.0)) is False
    assert los((0.0, 0.0), (1000.0, 1000.0)) is False


def test_feasible_grid_los_rejects_wrong_coordinate_contract(tmp_path):
    metadata = tmp_path / "feasible.json"
    metadata.write_text(json.dumps({"source1": {}}))
    arrays = tmp_path / "feasible.npz"
    np.savez_compressed(arrays, feasible_mask=np.ones((1, 1), dtype=np.uint8))
    with pytest.raises(ValueError, match="coordinate contract"):
        line_of_sight_from_feasible_grid({
            "metadata": str(metadata),
            "metadata_key": "source1",
            "arrays": str(arrays),
            "mask_key": "feasible_mask",
            "coordinate_contract": "room_specific_guess",
        })


def test_instant_distance_order_makes_allocated_target_closer():
    scene = synthetic_scene()
    ledger = RejectionLedger()
    plan = solve_instant_distance_order(
        scene, PARAMS, query_frame=30, profile_id="card4R",
        idle_choices=(0, 8), rng=np.random.default_rng(44), ledger=ledger,
        min_distance_gap_cm=50.0)
    assert not isinstance(plan, Rejection), ledger.summary()
    assert plan.checks["distance_gap_cm"] >= 50.0
    assert (plan.answer_cell["other_distance_cm"]
            > plan.answer_cell["target_distance_cm"])
    assert plan.target_route.route_id != plan.other_route.route_id


@pytest.mark.parametrize("relation", ["closer", "farther"])
def test_distance_change_pair_has_opposite_distractor_trend(relation):
    scene = synthetic_scene()
    ledger = RejectionLedger()
    plan = solve_distance_change_pair(
        scene, PARAMS, start_frame=12, end_frame=40,
        target_relation=relation, profile_id="card5R",
        idle_choices=(0, 8), rng=np.random.default_rng(55),
        ledger=ledger, min_change_cm=25.0)
    assert not isinstance(plan, Rejection), ledger.summary()
    target_delta = plan.answer_cell["target_delta_cm"]
    other_delta = plan.answer_cell["other_delta_cm"]
    assert (target_delta < -25.0 and other_delta > 25.0) if relation == "closer" \
        else (target_delta > 25.0 and other_delta < -25.0)


def test_route_pause_freezes_window_and_resumes_delayed_motion():
    route = straight_route("r", (0.0, 0.0), (740.0, 0.0))
    paused = route.paused(20, 30)
    assert paused.at(20) == paused.at(30)
    assert paused.at(31) == route.at(21)
    assert paused.displacement_cm > 0.0


@pytest.mark.parametrize("state", ["moving", "still"])
def test_motion_state_pair_has_opposite_window_states(state):
    scene = synthetic_scene()
    ledger = RejectionLedger()
    plan = solve_motion_state_pair(
        scene, PARAMS, start_frame=24, end_frame=40,
        target_state=state, profile_id="card6",
        idle_choices=(0, 8), rng=np.random.default_rng(66),
        ledger=ledger, min_motion_cm=10.0)
    assert not isinstance(plan, Rejection), ledger.summary()
    target = plan.answer_cell["target_window_displacement_cm"]
    other = plan.answer_cell["other_window_displacement_cm"]
    if state == "moving":
        assert target >= 10.0 and other <= 1.0e-6
    else:
        assert target <= 1.0e-6 and other >= 10.0
    assert plan.checks["uses_solved_route_samples_directly"] is True

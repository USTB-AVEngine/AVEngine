"""Route synthesis: a designed route is an ordinary route under the same checks.

合成场景(一间带中央岛台的方房)证明:合成路线在关键帧准确落进方位带与
距离范围、速度由反解保证落在声明范围、前后腿只在设计帧转向、实际占用的
每一帧都在可走区内;库预算用完求解器才合成,合成候选与库候选走完全相同的
检查并把来源记进 checks;开了合成却没有栅格就拒绝启动;同一种子结果可复现。
不引用任何真实房间。
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
    Rejection,
    RejectionLedger,
    Route,
    SceneInputs,
    attempt_budgets,
    circular_gap_deg,
    load_scene,
    relative_azimuth_deg,
    require_route_synthesis,
    route_synthesis_report,
    route_synthesizer,
    solve_backward_cross_time,
    solve_distance_change_pair,
    solve_forward_cross_time,
    solve_instant_azimuth,
    solve_instant_binding,
    solve_instant_distance_order,
    solve_motion_state_pair,
)
from route_synthesis import (  # noqa: E402
    REASON_SPEED,
    REASON_WALKABLE,
    PointSpec,
    RouteSynthesizer,
    SynthesisSettings,
    polyline_positions,
    solve_ray_distance,
)
from walkable_grid import WalkableGrid, write_walkable_grid  # noqa: E402
from build_qa_v3_walkable_grid import validate_against_scene  # noqa: E402

PARAMS = {"THETA_FULL": 15.0, "THETA_HALF": 30.0, "MIN_AZIMUTH_SEP": 25.0,
          "MIN_CAMERA_DISTANCE_CM": 100.0}
BANDS = [(-52.5, -17.5), (-17.5, 17.5), (17.5, 52.5)]
SYNTH = dict(PARAMS, ROUTE_SYNTHESIS_ENABLED=True,
             ROUTE_SYNTHESIS_SPEED_MPS_RANGE=[0.6, 1.5],
             ROUTE_SYNTHESIS_WALKABLE_MARGIN_M=0.3,
             ROUTE_SYNTHESIS_MAX_CAMERA_DISTANCE_CM=600.0,
             ROUTE_SYNTHESIS_ATTEMPTS=2500,
             ROUTE_SYNTHESIS_MAX_TURN_DEG=90.0)
ROOM_CM = 1400.0
CELL_CM = 10.0
ISLAND = (-100.0, -100.0, 100.0, 100.0)      # a 2 m furniture island in the middle


def room_grid(root, scene_id="synth_room"):
    n = int(ROOM_CM / CELL_CM)
    origin = (-ROOM_CM / 2.0, -ROOM_CM / 2.0)
    walkable = np.ones((n, n), dtype=bool)
    for row in range(n):
        for col in range(n):
            x = origin[0] + (col + 0.5) * CELL_CM
            y = origin[1] + (row + 0.5) * CELL_CM
            if ISLAND[0] <= x <= ISLAND[2] and ISLAND[1] <= y <= ISLAND[3]:
                walkable[row, col] = False
    write_walkable_grid(root, scene_id=scene_id, cell_cm=CELL_CM, origin_xy_cm=origin,
                        walkable=walkable, source={"kind": "feasible_region_mask", "test": True})
    return WalkableGrid.load(root)


def ring_points(radius=500.0, count=24):
    return [(radius * math.cos(2 * math.pi * k / count), radius * math.sin(2 * math.pi * k / count))
            for k in range(count)]


def straight_route(route_id, start, end, speed=0.7):
    samples = [(start[0] + (end[0] - start[0]) * i / (FRAME_COUNT - 1),
                start[1] + (end[1] - start[1]) * i / (FRAME_COUNT - 1))
               for i in range(FRAME_COUNT)]
    return Route(route_id, samples, speed)


def scene_with(grid, routes, scene_id="synth_room"):
    points = ring_points()
    return SceneInputs(scene_id=scene_id, backend="synthetic", routes=routes,
                       stand_points=points, camera_points=points, camera_height_m=1.47,
                       hfov_deg=105.0, walkable=grid,
                       provenance={"walkable_grid": grid.identity})


def short_bank_scene(grid):
    """Every bank route moves only 20 cm: no card1 sweep can come from the bank."""
    routes = []
    for k, (x, y) in enumerate(ring_points(radius=350.0)):
        routes.append(straight_route(f"short_{k}", (x, y), (x + 14.0, y + 14.0)))
    return scene_with(grid, routes)


def static_bank_scene(grid):
    routes = [Route(f"still_{k}", [xy] * FRAME_COUNT, 0.0)
              for k, xy in enumerate(ring_points(radius=350.0))]
    return scene_with(grid, routes)


def moving_bank_scene(grid):
    points = ring_points(radius=450.0)
    routes = [straight_route(f"r{k}", points[k], points[(k + 4) % len(points)])
              for k in range(len(points))]
    return scene_with(grid, routes)


# ---------------------------------------------------------------------------
# settings and fail-closed behaviour
# ---------------------------------------------------------------------------

def test_settings_fail_closed(tmp_path):
    assert SynthesisSettings.from_params(PARAMS) is None
    with pytest.raises(ValueError, match="lack"):
        SynthesisSettings.from_params(dict(PARAMS, ROUTE_SYNTHESIS_ENABLED=True))
    with pytest.raises(ValueError, match="SPEED"):
        SynthesisSettings.from_params(dict(SYNTH, ROUTE_SYNTHESIS_SPEED_MPS_RANGE=[1.5, 0.6]))
    with pytest.raises(ValueError, match="MARGIN"):
        SynthesisSettings.from_params(dict(SYNTH, ROUTE_SYNTHESIS_WALKABLE_MARGIN_M=-1))
    with pytest.raises(ValueError, match="TURN"):
        SynthesisSettings.from_params(dict(SYNTH, ROUTE_SYNTHESIS_MAX_TURN_DEG=200))
    settings = SynthesisSettings.from_params(SYNTH)
    assert settings.margin_cm == 30.0 and settings.synthesized_attempts == 2500
    assert settings.max_turn_deg == 90.0
    assert attempt_budgets(None, 300) == (300, 300)
    grid = room_grid(tmp_path / "grid")
    scene = short_bank_scene(grid)
    require_route_synthesis(scene, PARAMS)           # off: nothing to require
    require_route_synthesis(scene, SYNTH)
    assert attempt_budgets(route_synthesizer(scene, SYNTH), 300) == (300, 2800)
    scene.walkable = None
    with pytest.raises(ValueError, match="walkable_grid"):
        require_route_synthesis(scene, SYNTH)
    with pytest.raises(ValueError, match="walkable_grid"):
        route_synthesizer(scene, SYNTH)
    assert route_synthesizer(scene, PARAMS) is None
    report = route_synthesis_report(scene, PARAMS)
    assert report == {"enabled": False, "settings": None, "walkable_grid": grid.identity,
                      "order": "bank_only"}
    scene.walkable = grid
    with pytest.raises(ValueError, match="margin"):
        require_route_synthesis(scene, dict(SYNTH, ROUTE_SYNTHESIS_WALKABLE_MARGIN_M=50.0))


# ---------------------------------------------------------------------------
# the synthesizer itself
# ---------------------------------------------------------------------------

def test_ray_distance_solver_and_polyline_geometry():
    # camera at the origin, ray along +x, target at (300, 400): distance 500
    assert solve_ray_distance((0.0, 0.0), 0.0, (300.0, 400.0), 400.0) == [300.0]
    assert solve_ray_distance((0.0, 0.0), 0.0, (300.0, 400.0), 500.0) == [0.0, 600.0]
    assert solve_ray_distance((0.0, 0.0), 0.0, (300.0, 400.0), 100.0) == []
    # a straight two-point leg with idle un-shift
    line = polyline_positions([(0, (0.0, 0.0)), (10, (10.0, 0.0))], 1.0, 0.0, 0.0, 0)
    assert line[5] == (5.0, 0.0) and line[74] == (74.0, 0.0)
    assert polyline_positions([(0, (0.0, 0.0)), (10, (10.0, 0.0))], 1.0, 0.0, 0.0, 4)[0] == (4.0, 0.0)
    # one designed frame: the incoming leg may turn at that frame
    turned = polyline_positions([(10, (0.0, 0.0))], 1.0, 90.0, 0.0, 0)
    assert turned[5] == pytest.approx((0.0, -5.0))
    assert turned[15] == pytest.approx((5.0, 0.0))


def test_designed_route_passes_through_its_points_at_a_solved_speed(tmp_path):
    grid = room_grid(tmp_path / "grid")
    synth = RouteSynthesizer(grid, SynthesisSettings.from_params(SYNTH))
    camera, yaw = (-500.0, 0.0), 0.0
    rng = np.random.default_rng(7)
    specs = [PointSpec(40, -20.0, -10.0, 300.0, 400.0), PointSpec(74, 10.0, 20.0, 300.0, 400.0)]
    built = 0
    for _ in range(100):
        for idle in (0, 8, 16):
            route, reason = synth.design(rng, camera, yaw, specs, idle_frames=idle, role="target")
            if route is None:
                assert reason in (REASON_SPEED, REASON_WALKABLE)
                continue
            built += 1
            moved = route.shifted(idle)
            design = route.provenance["design"]
            assert design["points"][0]["solved"] is True and design["points"][1]["solved"] is False
            for point in design["points"]:
                xy = moved.at(point["frame"])
                assert math.dist(xy, point["xy_cm"]) < 1e-6
                azimuth = relative_azimuth_deg(camera, yaw, xy)
                assert abs(azimuth - point["azimuth_deg"]) < 1e-3      # record rounds to 4 decimals
                assert abs(math.dist(camera, xy) - point["distance_cm"]) < 1e-2
            assert 0.6 - 1e-3 <= design["speed_mps"] <= 1.5 + 1e-3
            steps = [math.dist(route.samples_xy[k], route.samples_xy[k + 1])
                     for k in range(FRAME_COUNT - 1)]
            assert max(steps) - min(steps) < 1e-6          # one speed on every leg
            assert abs(steps[0] * 15.0 / 100.0 - design["speed_mps"]) < 1e-3
            # headings: the middle leg joins the designed points; the legs before
            # and after turn by the recorded angles, and nowhere else
            def heading(a, b):
                return math.degrees(math.atan2(b[1] - a[1], b[0] - a[0]))
            first, last = design["points"][0]["frame"], design["points"][1]["frame"]
            mid = heading(moved.at(first), moved.at(last))
            assert circular_gap_deg(mid, design["heading_deg"]) < 1e-3
            if first - 1 > idle:
                incoming = heading(moved.at(first - 1), moved.at(first))
                assert circular_gap_deg(incoming, mid + design["turn_in_deg"]) < 1e-3
                assert abs(design["turn_in_deg"]) <= 90.0
            if last < FRAME_COUNT - 1:
                outgoing = heading(moved.at(last), moved.at(last + 1))
                assert circular_gap_deg(outgoing, mid + design["turn_out_deg"]) < 1e-3
            assert route.source == "synthesized" and route.route_id.startswith("synth:")
            assert moved.source == "synthesized"           # provenance survives the shift
            for xy in moved.samples_xy:                    # every occupied frame keeps the margin
                assert grid.is_walkable(xy, margin_cm=30.0)
            assert design["min_clearance_cm"] >= 30.0
            assert design["checked_frames"] == FRAME_COUNT - idle
    assert built >= 8          # the island next to the designed points refuses many legs
    # a leg forced across the island: nothing is built and the grid says why
    across = [PointSpec(0, -0.5, 0.5, 250.0, 350.0), PointSpec(74, -0.5, 0.5, 650.0, 750.0)]
    reasons = set()
    for _ in range(40):
        route, reason = synth.design(rng, camera, yaw, across, idle_frames=0, role="target")
        assert route is None
        reasons.add(reason)
    assert REASON_WALKABLE in reasons
    # no distance on the anchor ray gives the declared speed: refused before any walk
    fast = [PointSpec(0, 30.0, 31.0, 200.0, 200.0), PointSpec(10, -31.0, -30.0, 200.0, 200.0)]
    route, reason = synth.design(np.random.default_rng(1), camera, yaw, fast,
                                 idle_frames=0, role="target")
    assert route is None and reason == REASON_SPEED
    assert synth.counters["rejected"][REASON_SPEED] >= 1


def test_one_point_design_walks_through_the_designed_frame(tmp_path):
    grid = room_grid(tmp_path / "grid")
    synth = RouteSynthesizer(grid, SynthesisSettings.from_params(SYNTH))
    rng = np.random.default_rng(11)
    route, reason = synth.design_many(rng, (-500.0, 0.0), 0.0,
                                      [PointSpec(30, -10.0, 10.0, 200.0, 250.0)],
                                      idle_frames=8, role="target", tries=64)
    assert route is not None, reason
    moved = route.shifted(8)
    design = route.provenance["design"]
    assert math.dist(moved.at(30), design["points"][0]["xy_cm"]) < 1e-6
    outgoing = math.degrees(math.atan2(moved.at(60)[1] - moved.at(30)[1],
                                       moved.at(60)[0] - moved.at(30)[0]))
    assert circular_gap_deg(outgoing, design["heading_deg"] + design["turn_out_deg"]) < 1e-3
    incoming = math.degrees(math.atan2(moved.at(30)[1] - moved.at(10)[1],
                                       moved.at(30)[0] - moved.at(10)[0]))
    assert circular_gap_deg(incoming, design["heading_deg"] + design["turn_in_deg"]) < 1e-3
    assert abs(design["turn_in_deg"]) <= 90.0 and abs(design["turn_out_deg"]) <= 90.0
    for xy in moved.samples_xy:
        assert grid.is_walkable(xy, margin_cm=30.0)


def test_exclusions_and_joint_gap_shape_the_draws(tmp_path):
    from route_synthesis import draw_from_intervals, subtract_windows
    assert subtract_windows((-52.5, -17.5), [(-40.0, 5.0)]) == [(-52.5, -45.0), (-35.0, -17.5)]
    assert subtract_windows((0.0, 10.0), [(5.0, 10.0)]) == []
    spec = PointSpec(30, -52.5, 52.5, 100.0, 300.0, exclusions=((0.0, 25.25),))
    assert spec.azimuth_intervals() == [(-52.5, -25.25), (25.25, 52.5)]
    rng = np.random.default_rng(3)
    for _ in range(200):
        value = draw_from_intervals(rng, spec.azimuth_intervals())
        assert abs(value) >= 25.25 and abs(value) <= 52.5
    grid = room_grid(tmp_path / "grid")
    synth = RouteSynthesizer(grid, SynthesisSettings.from_params(SYNTH))
    camera, yaw = (-500.0, 0.0), 0.0
    # anchor band and answer band overlap; a 30.25 degree sweep must still hold
    specs = [PointSpec(40, -52.5, -29.17, 100.0, 600.0), PointSpec(74, -52.5, -17.5, 100.0, 600.0)]
    built = 0
    for _ in range(300):
        route, reason = synth.design(rng, camera, yaw, specs, idle_frames=0, role="target",
                                     min_gap_between_points_deg=30.25)
        if route is None:
            continue
        built += 1
        points = route.provenance["design"]["points"]
        assert circular_gap_deg(points[0]["azimuth_deg"], points[1]["azimuth_deg"]) >= 30.25 - 1e-3
        assert route.provenance["design"]["min_gap_between_points_deg"] == 30.25
    assert built > 0
    # a gap no pair of azimuths in these bands can reach is refused as infeasible
    from route_synthesis import REASON_SPEC
    route, reason = synth.design(rng, camera, yaw, specs, idle_frames=0, role="target",
                                 min_gap_between_points_deg=40.0)
    assert route is None and reason == REASON_SPEC


# ---------------------------------------------------------------------------
# solvers: bank first, then synthesis, identical checks
# ---------------------------------------------------------------------------

def _verify_card1_plan(plan, scene, anchor_band, answer_band, anchor_frame, query_frame):
    camera, yaw = plan.camera_xy, plan.camera_ue_yaw_deg
    target = plan.target_route
    az_anchor = relative_azimuth_deg(camera, yaw, target.at(anchor_frame))
    az_query = relative_azimuth_deg(camera, yaw, target.at(query_frame))
    assert anchor_band[0] <= az_anchor < anchor_band[1]
    assert answer_band[0] <= az_query < answer_band[1]
    assert circular_gap_deg(az_anchor, az_query) > 30.0
    assert math.dist(camera, target.at(anchor_frame)) >= 100.0
    assert math.dist(camera, target.at(query_frame)) >= 100.0
    other = plan.other_route
    az_other_anchor = relative_azimuth_deg(camera, yaw, other.at(anchor_frame))
    az_other_query = relative_azimuth_deg(camera, yaw, other.at(query_frame))
    assert abs(az_other_anchor) <= 52.5 and abs(az_other_query) <= 52.5
    assert circular_gap_deg(az_other_anchor, az_anchor) >= 25.0
    assert not (answer_band[0] <= az_other_query < answer_band[1])
    assert circular_gap_deg(az_query, az_other_query) > 60.0
    sources = plan.checks["route_sources"]
    assert sources["target"] in ("bank", "synthesized")
    assert sources["other"] in ("bank", "synthesized")
    for role, route in (("target", plan.target_route), ("other", other)):
        if sources[role] == "synthesized":
            for xy in route.samples_xy:                    # occupied frames of the shifted route
                assert scene.walkable.is_walkable(xy, margin_cm=30.0)
            assert 0.6 - 1e-3 <= route.provenance["design"]["speed_mps"] <= 1.5 + 1e-3
            assert route.provenance["role"] == role


def test_card1_solvers_synthesize_only_after_the_bank_budget_and_re_verify(tmp_path):
    grid = room_grid(tmp_path / "grid")
    scene = short_bank_scene(grid)
    # without synthesis the short bank cannot fill any anchor x answer cell
    ledger = RejectionLedger()
    outcome = solve_forward_cross_time(scene, PARAMS, answer_band=BANDS[2], answer_bands=BANDS,
                                       anchor_frame=40, anchor_band=BANDS[0],
                                       idle_choices=(0, 8, 16),
                                       rng=np.random.default_rng(1), ledger=ledger,
                                       max_attempts=400)
    assert isinstance(outcome, Rejection)
    assert ledger.summary()["route_synthesis"]["synthesized_attempts"] == 0
    assert ledger.summary()["route_synthesis"]["bank_attempts"] == 400
    # with synthesis the bank keeps its whole budget, then every joint cell fills
    # from a designed target
    ledger = RejectionLedger()
    rng = np.random.default_rng(1)
    for anchor_band in BANDS:
        for answer_band in BANDS:
            plan = solve_forward_cross_time(scene, SYNTH, answer_band=answer_band,
                                            answer_bands=BANDS, anchor_frame=40,
                                            anchor_band=anchor_band, idle_choices=(0, 8, 16),
                                            rng=rng, ledger=ledger, max_attempts=500)
            assert not isinstance(plan, Rejection), (anchor_band, answer_band, ledger.summary())
            assert plan.checks["route_sources"]["target"] == "synthesized"
            assert plan.checks["search_attempts"] > 500
            _verify_card1_plan(plan, scene, anchor_band, answer_band, 40, 74)
            plan_b = solve_backward_cross_time(scene, SYNTH, answer_band=answer_band,
                                               answer_bands=BANDS, anchor_frame=62, query_frame=22,
                                               anchor_band=anchor_band, idle_choices=(0, 8),
                                               rng=rng, ledger=ledger, max_attempts=500)
            assert not isinstance(plan_b, Rejection), (anchor_band, answer_band, ledger.summary())
            assert plan_b.checks["route_sources"]["target"] == "synthesized"
            _verify_card1_plan(plan_b, scene, anchor_band, answer_band, 62, 22)
    synthesis = ledger.summary()["route_synthesis"]
    assert synthesis["bank_attempts"] == 18 * 500
    assert 0 < synthesis["synthesized_attempts"] <= 18 * 2500
    assert synthesis["target_built"] > 0 and synthesis["target_designs"] >= synthesis["target_built"]


def test_bank_routes_are_used_when_the_bank_suffices(tmp_path):
    grid = room_grid(tmp_path / "grid")
    scene = moving_bank_scene(grid)
    ledger = RejectionLedger()
    rng = np.random.default_rng(5)
    for _ in range(6):
        plan = solve_forward_cross_time(scene, SYNTH, answer_band=BANDS[1], answer_bands=BANDS,
                                        anchor_frame=45, idle_choices=(0, 8, 16), rng=rng,
                                        ledger=ledger, max_attempts=3000)
        assert not isinstance(plan, Rejection), ledger.summary()
        assert plan.checks["route_sources"]["target"] == "bank"
        assert plan.base_route.source_record == {"source": "bank", "route_id": plan.base_route.route_id}
        assert plan.checks["route_sources"]["other"] in ("bank", "synthesized")
    summary = ledger.summary()["route_synthesis"]
    assert summary["synthesized_attempts"] == 0 and summary["target_designs"] == 0


def test_instant_family_solvers_synthesize_when_the_bank_is_static(tmp_path):
    grid = room_grid(tmp_path / "grid")
    scene = static_bank_scene(grid)
    ledger = RejectionLedger()
    rng = np.random.default_rng(9)
    plan = solve_instant_azimuth(scene, SYNTH, answer_band=BANDS[0], answer_bands=BANDS,
                                 query_frame=30, profile_id="card2", idle_choices=(0, 8),
                                 rng=rng, ledger=ledger, max_attempts=300)
    assert not isinstance(plan, Rejection), ledger.summary()
    assert plan.checks["route_sources"] == {"target": "synthesized", "other": "synthesized"}
    azimuth = relative_azimuth_deg(plan.camera_xy, plan.camera_ue_yaw_deg, plan.target_route.at(30))
    assert BANDS[0][0] <= azimuth < BANDS[0][1]
    other_az = relative_azimuth_deg(plan.camera_xy, plan.camera_ue_yaw_deg, plan.other_route.at(30))
    assert circular_gap_deg(azimuth, other_az) >= 25.0 and abs(other_az) <= 52.5

    plan = solve_instant_binding(scene, SYNTH, instants=(12, 40), profile_id="card9",
                                 idle_choices=(0, 8), rng=rng, ledger=ledger, max_attempts=300)
    assert not isinstance(plan, Rejection), ledger.summary()
    assert plan.checks["route_sources"]["target"] == "synthesized"
    assert plan.checks["min_separation_deg"] >= 25.0

    plan = solve_instant_distance_order(scene, SYNTH, query_frame=30, profile_id="card4R",
                                        idle_choices=(0, 8), rng=rng, ledger=ledger,
                                        min_distance_gap_cm=50.0, max_attempts=300)
    assert not isinstance(plan, Rejection), ledger.summary()
    assert plan.checks["route_sources"]["target"] == "synthesized"
    assert plan.checks["distance_gap_cm"] >= 50.0

    for relation in ("closer", "farther"):
        plan = solve_distance_change_pair(scene, SYNTH, start_frame=40, end_frame=74,
                                          target_relation=relation, profile_id="card5R",
                                          idle_choices=(0, 8, 16), rng=rng, ledger=ledger,
                                          min_change_cm=50.0, max_attempts=300)
        assert not isinstance(plan, Rejection), (relation, ledger.summary())
        assert plan.checks["route_sources"]["target"] == "synthesized"
        delta = plan.checks["target_distance_delta_cm"]
        assert (delta <= -50.0) if relation == "closer" else (delta >= 50.0)
        other_delta = plan.checks["other_distance_delta_cm"]
        assert (other_delta >= 50.0) if relation == "closer" else (other_delta <= -50.0)

    for state in ("moving", "still"):
        plan = solve_motion_state_pair(scene, SYNTH, start_frame=29, end_frame=74,
                                       target_state=state, profile_id="card6R",
                                       idle_choices=(0, 8, 16), rng=rng, ledger=ledger,
                                       min_motion_cm=30.0, max_attempts=300)
        assert not isinstance(plan, Rejection), (state, ledger.summary())
        assert plan.checks["route_sources"]["target"] == "synthesized"
        window = math.dist(plan.target_route.at(29), plan.target_route.at(74))
        assert (window >= 30.0) if state == "moving" else (window <= 1e-6)
        assert plan.target_route.displacement_cm > 0.0     # the still actor still moves outside the window
        other_window = math.dist(plan.other_route.at(29), plan.other_route.at(74))
        assert (other_window <= 1e-6) if state == "moving" else (other_window >= 30.0)
    counters = ledger.summary()["route_synthesis"]
    assert counters["other_built"] > 0 and counters["target_built"] > 0


def test_synthesis_is_deterministic_for_a_seed(tmp_path):
    grid = room_grid(tmp_path / "grid")
    scene = short_bank_scene(grid)
    plans = []
    for _ in range(2):
        plan = solve_forward_cross_time(scene, SYNTH, answer_band=BANDS[2], answer_bands=BANDS,
                                        anchor_frame=40, anchor_band=BANDS[0],
                                        idle_choices=(0, 8, 16), rng=np.random.default_rng(21),
                                        ledger=RejectionLedger(), max_attempts=200)
        assert not isinstance(plan, Rejection)
        plans.append(plan)
    assert plans[0].base_route.route_id == plans[1].base_route.route_id
    assert plans[0].other_route.route_id == plans[1].other_route.route_id
    assert plans[0].camera_xy == plans[1].camera_xy
    assert plans[0].camera_ue_yaw_deg == plans[1].camera_ue_yaw_deg


def test_ledgers_fold_synthesis_counters(tmp_path):
    grid = room_grid(tmp_path / "grid")
    scene = short_bank_scene(grid)
    first, second, total = RejectionLedger(), RejectionLedger(), RejectionLedger()
    for ledger in (first, second):
        plan = solve_forward_cross_time(scene, SYNTH, answer_band=BANDS[2], answer_bands=BANDS,
                                        anchor_frame=40, anchor_band=BANDS[0],
                                        idle_choices=(0, 8, 16), rng=np.random.default_rng(2),
                                        ledger=ledger, max_attempts=100)
        assert not isinstance(plan, Rejection)
        total.absorb(ledger)
    summary = total.summary()
    assert summary["route_synthesis"]["bank_attempts"] == 200
    assert summary["combinations_evaluated"] == (first.combinations_evaluated
                                                 + second.combinations_evaluated)
    assert summary["route_synthesis"]["target_built"] == 2 * first.synthesis["target_built"]
    assert summary["total"] == sum(first.counts.values()) + sum(second.counts.values())


# ---------------------------------------------------------------------------
# scene loading
# ---------------------------------------------------------------------------

def _write_bank_and_config(tmp_path, grid_dir, scene_id, routes, name="bank.json"):
    bank = tmp_path / name
    bank.write_text(json.dumps({"schema": "avengine_apartment_route_bank_v1", "routes": [
        {"route_id": r.route_id, "samples_ue_cm": [list(p) for p in r.samples_xy],
         "implied_speed_mps": 0.7} for r in routes]}))
    request = tmp_path / "request.json"
    request.write_text(json.dumps({
        "primary_camera_rig": {"world_from_rig": {"translation_m": [0, 1.47, 0]},
                               "shared_calibration": {"hfov_degrees": 105.0}},
        "listener": {"rig_from_listener": {"translation_m": [0, 0, 0]}}}))
    return {"scene_id": scene_id, "backend": "synthetic", "route_bank": str(bank),
            "camera_base_request": str(request), "walkable_grid": str(grid_dir)}


def test_load_scene_reads_the_grid_and_rejects_a_grid_that_does_not_fit(tmp_path):
    grid_dir = tmp_path / "grid"
    room_grid(grid_dir, scene_id="room_a")
    routes = [straight_route(f"r{k}", (300.0 + 10 * k, -200.0), (350.0 + 10 * k, 200.0))
              for k in range(4)]
    config = _write_bank_and_config(tmp_path, grid_dir, "room_a", routes)
    scene = load_scene(config)
    assert scene.walkable is not None and scene.route_synthesis_available
    assert scene.provenance["walkable_grid"]["scene_id"] == "room_a"
    assert scene.provenance["walkable_grid"]["arrays_sha256"]
    with pytest.raises(ValueError, match="belongs to"):
        load_scene(dict(config, scene_id="room_b"))
    # a route ending on the island puts a camera point off the grid: refuse to load
    island_routes = routes + [straight_route("r_island", (-400.0, 0.0), (0.0, 0.0))]
    with pytest.raises(ValueError, match="does not contain"):
        load_scene(_write_bank_and_config(tmp_path, grid_dir, "room_a", island_routes,
                                          name="bank_island.json"))
    plain = dict(config)
    del plain["walkable_grid"]
    scene = load_scene(plain)
    assert scene.walkable is None and scene.provenance["walkable_grid"] is None
    validation = validate_against_scene(WalkableGrid.load(grid_dir), scene, [0.3])
    assert validation["bank_samples_inside_fraction"] == 1.0
    assert validation["camera_points_inside_fraction"] == 1.0
    assert set(validation["by_margin_m"]) == {"0.3"}

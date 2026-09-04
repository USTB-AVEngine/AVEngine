"""F2 off-screen/rear query geometry and wrap-aware arc tests."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

TOOLS = Path(__file__).resolve().parents[1] / "tools" / "qa"
sys.path.insert(0, str(TOOLS))

import qa_v3_azimuth as AZ  # noqa: E402
from qa_v3_arc import Arc  # noqa: E402
import scene_sampler as SS  # noqa: E402


PARAMS = {
    "THETA_FULL": 15.0,
    "THETA_HALF": 30.0,
    "MIN_AZIMUTH_SEP": 25.0,
    "MIN_CAMERA_DISTANCE_CM": 100.0,
}


def _polar(degrees, radius=300.0):
    angle = math.radians(degrees)
    return radius * math.cos(angle), radius * math.sin(angle)


def _route(route_id, anchor_degrees, query_degrees):
    anchor = _polar(anchor_degrees)
    query = _polar(query_degrees)
    samples = [anchor] * 46
    for frame in range(46, SS.FRAME_COUNT):
        fraction = (frame - 45) / (SS.FRAME_COUNT - 1 - 45)
        samples.append((
            anchor[0] + fraction * (query[0] - anchor[0]),
            anchor[1] + fraction * (query[1] - anchor[1]),
        ))
    return SS.Route(route_id, samples, 0.7)


def _scene(target_query=180.0, other_query=20.0):
    return SS.SceneInputs(
        scene_id="synthetic_f2",
        backend="synthetic",
        routes=[
            _route("target", 0.0, target_query),
            _route("other", 70.0, other_query),
        ],
        stand_points=[(0.0, 0.0)],
        camera_points=[(0.0, 0.0)],
        camera_height_m=1.47,
        hfov_deg=180.0,
    )


def test_arc_from_samples_keeps_negative_long_sweep_and_forward_wrap():
    arc = Arc.from_samples([100.0, 50.0, 0.0, -50.0, -100.0])
    assert arc.start_deg == pytest.approx(100.0)
    assert arc.sweep_deg == pytest.approx(-200.0)
    assert arc.width_deg == pytest.approx(200.0)
    assert arc.contains(0.0)
    assert not arc.contains(120.0)

    wrapped = Arc.from_forward_bounds(170.0, -170.0)
    assert wrapped.start_deg == pytest.approx(170.0)
    assert wrapped.sweep_deg == pytest.approx(20.0)
    assert wrapped.contains(179.0)
    assert wrapped.contains(-179.0)
    assert not wrapped.contains(0.0)


def test_published_arc_conversion_preserves_seam_and_direction():
    engine = Arc(start_deg=170.0, sweep_deg=10.0)
    published = AZ.to_published_arc(engine)
    assert published.start_deg == pytest.approx(-170.0)
    assert published.sweep_deg == pytest.approx(-10.0)
    assert published.contains(-175.0)
    assert not published.contains(0.0)


def test_rear_cone_odd_band_count_is_exact_and_count_one_is_allowed():
    scene = _scene()
    profile = {
        "id": "f2_rear",
        "answer_domain": "rear_cone",
        "answer_shape": {"equal_bands": 3},
    }
    bands = SS.derive_answer_bands(profile, scene, {})
    assert len(bands) == 3
    assert all(SS.band_width_deg(band) == pytest.approx(60.0)
               for band in bands)
    single = SS.derive_answer_bands({
        "id": "f2_single",
        "answer_domain": "rear_cone",
        "answer_shape": {"equal_bands": 1},
    }, scene, {})
    assert len(single) == 1
    assert single[0].width_deg == pytest.approx(180.0)


def test_full_circle_solver_can_return_a_behind_offscreen_query():
    scene = _scene(other_query=20.0)
    bands = [(-180.0, -90.0), (-90.0, 0.0), (0.0, 90.0), (90.0, 180.0)]
    ledger = SS.RejectionLedger()
    plan = SS.solve_forward_cross_time(
        scene, PARAMS,
        answer_band=(135.0, 180.0), answer_bands=bands,
        anchor_frame=45, idle_choices=(0,), rng=np.random.default_rng(11),
        ledger=ledger, max_attempts=200,
        query_domain="full_circle",
    )
    assert not isinstance(plan, SS.Rejection), ledger.summary()
    assert 135.0 <= plan.answer_cell["value_deg"] < 180.0
    assert abs(plan.answer_cell["value_deg"]) > scene.hfov_deg / 2.0
    assert plan.checks["query_bound_deg"] == pytest.approx(180.0)
    assert plan.checks["query_requires_visibility"] is False
    assert abs(SS.relative_azimuth_deg(
        plan.camera_xy, plan.camera_ue_yaw_deg,
        plan.other_route.at(plan.query_frame))) <= 180.0


def test_rear_cone_solver_can_return_a_back_answer_with_two_sides():
    scene = _scene(other_query=-105.0)
    profile = {
        "id": "f2_rear_even",
        "answer_domain": "rear_cone",
        "answer_shape": {"equal_bands": 4},
    }
    bands = SS.derive_answer_bands(profile, scene, {})
    answer_band = bands[-1]
    ledger = SS.RejectionLedger()
    plan = SS.solve_forward_cross_time(
        scene, PARAMS,
        answer_band=answer_band, answer_bands=bands,
        anchor_frame=45, idle_choices=(0,), rng=np.random.default_rng(13),
        ledger=ledger, max_attempts=300,
        answer_domain="rear_cone",
    )
    assert not isinstance(plan, SS.Rejection), ledger.summary()
    assert SS.band_to_arc(answer_band).contains(plan.answer_cell["value_deg"])
    assert abs(plan.answer_cell["value_deg"]) > scene.hfov_deg / 2.0
    assert plan.checks["query_bound_deg"] == pytest.approx(180.0)
    assert plan.checks["query_requires_visibility"] is False
    assert plan.checks["gatea_open_gold_separation_deg"] > 2 * PARAMS["THETA_HALF"]


def test_default_solver_query_geometry_keeps_visible_front_behavior():
    bound, visible = SS.resolve_query_geometry(_scene(), PARAMS)
    assert bound == pytest.approx(90.0)
    assert visible is True
    bound, visible = SS.resolve_query_geometry(
        _scene(), PARAMS, query_bound_deg=60.0,
        query_requires_visibility=True)
    assert (bound, visible) == (60.0, True)

"""Tests for deterministic, suffix-sensitive N-actor scene search."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

TOOLS = Path(__file__).resolve().parents[1] / "tools" / "qa"
sys.path.insert(0, str(TOOLS))

from build_qa_v3_n_actor_canary import (  # noqa: E402
    NRouteSearchExhausted,
    find_n_route_plan,
    seed_uint64,
)
from scene_sampler import Route, SceneInputs  # noqa: E402


def _route(route_id, angle_deg):
    angle = math.radians(angle_deg)
    samples = []
    for frame in range(75):
        radius = 500.0 + frame
        samples.append((radius * math.cos(angle), radius * math.sin(angle)))
    return Route(route_id, samples, 0.2)


def _scene(*, hfov=179.0):
    return SceneInputs(
        scene_id="synthetic",
        backend="test",
        routes=[_route(f"route_{index}", index * 45.0) for index in range(8)],
        stand_points=[(0.0, 0.0)],
        camera_points=[(0.0, 0.0)],
        camera_height_m=1.5,
        hfov_deg=hfov,
    )


def _signature(plan):
    return (
        tuple(plan["camera_xy"]),
        round(float(plan["camera_yaw_deg"]), 9),
        tuple(route.route_id for route in plan["routes"]),
    )


def test_complete_seed_suffix_changes_rng_entropy_and_plan():
    seed_a = "qa-v3-common-prefix|card17|segment2|0"
    seed_b = "qa-v3-common-prefix|card17|segment2|1"
    assert seed_uint64(seed_a) != seed_uint64(seed_b)
    plan_a = find_n_route_plan(
        _scene(), {}, actor_count=2, seed=seed_a,
        binding_frames=(12, 40), min_pairwise_sep_deg=5.0)
    plan_b = find_n_route_plan(
        _scene(), {}, actor_count=2, seed=seed_b,
        binding_frames=(12, 40), min_pairwise_sep_deg=5.0)
    assert _signature(plan_a) != _signature(plan_b)


def test_failed_search_reports_true_evaluated_denominator():
    with pytest.raises(NRouteSearchExhausted) as captured:
        find_n_route_plan(
            _scene(hfov=1.0), {}, actor_count=2,
            seed="qa-v3-failure-denominator",
            binding_frames=(12, 40), max_attempts=17)
    assert captured.value.evaluated_combinations == 17

from __future__ import annotations

from copy import deepcopy

import pytest

from avengine.rooms import conditioned_visibility as cv
from avengine.rooms.target_observability import (
    DEFAULT_MIN_PROJECTED_AREA_PX,
    rank_projected_point_rows,
    score_projected_target_point,
    score_screen_visibility_frame,
)


def _frame(
    *,
    state: str = "visible_clear",
    bbox: list[float] | None = None,
    border_marginal_samples: int = 0,
) -> dict:
    return {
        "state": state,
        "projected_body_bbox_px": (
            [10.0, 10.0, 50.0, 30.0] if bbox is None else bbox
        ),
        "border_marginal_samples": border_marginal_samples,
    }


def _camera() -> cv.CameraPose:
    return cv.CameraPose(
        candidate_id="point",
        position_m=(0.0, 1.55, 0.0),
        forward=(0.0, 0.0, -1.0),
        right=(1.0, 0.0, 0.0),
        up=(0.0, 1.0, 0.0),
        horizontal_fov_deg=85.0,
        resolution_hw=(720, 1280),
    )


def test_screen_frame_uses_existing_bbox_border_and_state_fields() -> None:
    result = score_screen_visibility_frame(_frame(), [100, 100])
    assert result["eligible"] is True
    assert result["projected_area_px"] == 800.0
    assert result["projected_observability_score"] == 800.0 / 512.0
    assert result["state"] == "visible_clear"

    assert score_screen_visibility_frame(
        _frame(border_marginal_samples=1), [100, 100]
    )["reason"] == "border_marginal"
    assert score_screen_visibility_frame(
        _frame(state="fully_occluded"), [100, 100]
    )["reason"] == "state_not_visible"
    assert score_screen_visibility_frame(
        _frame(bbox=[-1.0, 10.0, 50.0, 30.0]), [100, 100]
    )["reason"] == "projected_bbox_touches_frame_edge"
    assert score_screen_visibility_frame(
        _frame(bbox=[10.0, 10.0, 30.0, 30.0]), [100, 100]
    )["reason"] == "projected_area_below_minimum"


def test_point_score_reuses_existing_screen_projection() -> None:
    result = score_projected_target_point(
        camera=_camera(),
        root_m=(0.0, 0.0, -4.0),
        body=cv.BodyProxy(height_m=1.8, width_m=0.4),
    )
    assert result["in_view"] is True
    assert result["projected_body_bbox_px"] is not None
    assert result["projected_area_px"] >= DEFAULT_MIN_PROJECTED_AREA_PX
    assert result["eligible"] is True
    assert "native pixel" in result["claim_boundary"]


def test_unusable_point_is_zero_score_and_not_rejected() -> None:
    result = score_projected_target_point(
        camera=_camera(),
        root_m=(0.0, 0.0, 4.0),
        body=cv.BodyProxy(height_m=1.8, width_m=0.4),
    )
    assert result["in_view"] is False
    assert result["projected_observability_score"] == 0.0
    assert result["eligible"] is False


def test_point_rows_are_sorted_before_existing_budget_and_preserved() -> None:
    rows = [
        {"point": "low", "projected_observability_score": 0.2},
        {"point": "high", "projected_observability_score": 2.0},
        {"point": "missing"},
    ]
    before = deepcopy(rows)
    ranked = rank_projected_point_rows(rows, limit=2)
    assert [row["point"] for row in ranked] == ["high", "low"]
    assert rows == before
    all_rows = rank_projected_point_rows(rows)
    assert [row["point"] for row in all_rows] == ["high", "low", "missing"]


def test_preference_is_limited_to_explicit_appearance_targets():
    from avengine.rooms.conditioned_sampler import _appearance_target_ids
    compiled = [
        {'qa_id': 'QA-07', 'conditions': [
            {'kind': 'appearance_reference', 'subject': 'target'},
            {'kind': 'visibility_state', 'subject': 'other'}]},
        {'qa_id': 'QA-09', 'conditions': [
            {'kind': 'appearance_reference', 'subject': 'unrequested'}]},
    ]
    assert _appearance_target_ids({'qa_targets': [{'qa_id': 'QA-07'}]}, compiled) == {'target'}
    assert _appearance_target_ids({'qa_ids': ['QA-07']}, compiled) == set()


def test_cloud_preference_adds_projection_only_when_requested():
    import numpy as np
    from avengine.rooms.conditioned_sampler import _classify_cloud
    args = (_camera(), np.array([[0., 0., -4.]]),
            cv.BodyProxy(height_m=1.8, width_m=.4), [0., 1.6, 0.],
            None, cv.screen_policy(None), {})
    ordinary = _classify_cloud(*args, max_distance_m=6, cast_rays=False)
    preferred = _classify_cloud(*args, max_distance_m=6, cast_rays=False,
                                target_observability=True)
    assert ordinary[0]['state'] == preferred[0]['state']
    assert 'projected_observability_score' not in ordinary[0]
    assert preferred[0]['projected_observability_score'] > 0


@pytest.mark.parametrize("root", [(0., 0., -4.), (3., 0., -4.)])
def test_point_state_keeps_ray_results_and_cache_after_mesh_narrowing(monkeypatch, root):
    import numpy as np
    from avengine.qa.answerability import MeshHandle
    from avengine.rooms.conditioned_sampler import _point_state
    camera = _camera()
    vertices = np.array([[-1., -1., -2.], [1., -1., -2.], [1., 3., -2.], [-1., 3., -2.],
                         [-1., -1., 20.], [1., -1., 20.], [1., 3., 20.], [-1., 3., 20.]])
    triangles = np.array([[0, 1, 2], [0, 2, 3], [4, 5, 6], [4, 6, 7]])
    mesh = MeshHandle(vertices, triangles, {})
    body = cv.BodyProxy(height_m=1.8, width_m=.4)
    policy = cv.screen_policy(None)
    origin = np.asarray(camera.position_m)
    samples = cv.body_sample_points(origin, root, body, policy)
    baseline_cache = {}
    blocked, measured = cv._blocked_by_scene(mesh, origin, samples, policy, baseline_cache)
    inside = cv._project(camera, samples, policy)['in_frustum']
    expected = 'hidden' if blocked[inside].all() else 'visible' if not blocked[inside].any() else 'partial'
    assert measured[inside].all()
    calls = []
    ray = cv.line_of_sight
    def record(selected, source, endpoint):
        calls.append(len(selected.triangles))
        return ray(selected, source, endpoint)
    monkeypatch.setattr(cv, 'line_of_sight', record)
    actual_cache = {}
    actual, _ = _point_state(camera, np.asarray(root), body, mesh, policy, actual_cache)
    assert actual == expected
    assert actual_cache == baseline_cache
    assert len(calls) == len(baseline_cache)
    assert all(count < len(mesh.triangles) for count in calls)


@pytest.mark.parametrize("qa_id,branch", [("QA-07", "left"), ("QA-09", "yes"),
                                         ("QA-10", None), ("QA-11", None)])
def test_real_transition_translation_enables_the_named_visual_reference(qa_id, branch):
    from avengine.qa import generation_conditions as gc
    from avengine.rooms.conditioned_sampler import _appearance_target_ids
    conditions = gc._HANDLERS[qa_id](
        subjects=(gc.ConditionSubject(entity_instance_id='named_target', role='target'),),
        branch=branch, precision=0)
    payload = {'qa_id': qa_id, 'conditions': [condition.to_dict() for condition in conditions]}
    assert _appearance_target_ids({'qa_targets': [{'qa_id': qa_id}]}, [payload]) == {'named_target'}
    assert _appearance_target_ids({'qa_ids': [qa_id]}, [payload]) == set()

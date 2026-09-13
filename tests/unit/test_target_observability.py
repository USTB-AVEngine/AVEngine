from __future__ import annotations

from copy import deepcopy

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

from __future__ import annotations

import numpy as np
import pytest

from avengine.assets.glb import GlbDocument
from avengine.assets.world_contact import (
    WorldContactError,
    evaluate_idle_contact_gate,
    evaluate_walk_dynamic_gate,
    fit_constant_root_step,
    infer_uniform_skin_linear_scale,
    infer_height_backward_stance,
)


def _paw_cycle() -> np.ndarray:
    # Low/rearward stance occupies indices 1..4; the raised return stroke is
    # the forward swing and must never be relabelled as floor contact.
    return np.asarray(
        [
            [0.08, 0.03, 0.0],
            [0.06, 0.00, 0.0],
            [0.04, 0.00, 0.0],
            [0.02, 0.00, 0.0],
            [0.00, 0.00, 0.0],
            [0.01, 0.05, 0.0],
            [0.04, 0.08, 0.0],
            [0.07, 0.06, 0.0],
        ],
        dtype=np.float64,
    )


def _skin_document(scale: float, *, mismatched_last_bone: bool = False) -> GlbDocument:
    nodes = []
    for index in range(5):
        node: dict[str, object] = {"name": f"joint_{index}"}
        if index < 4:
            node["children"] = [index + 1]
        if index:
            bone_scale = scale * (1.2 if mismatched_last_bone and index == 4 else 1.0)
            node["translation"] = [bone_scale, 0.0, 0.0]
        nodes.append(node)
    return GlbDocument(
        json={
            "asset": {"version": "2.0"},
            "nodes": nodes,
            "skins": [{"skeleton": 0, "joints": list(range(5))}],
        },
        binary=b"",
        sha256=("a" if scale == 1.0 else "b") * 64,
        byte_length=0,
    )


def test_stance_requires_low_height_and_rearward_motion() -> None:
    states = infer_height_backward_stance(_paw_cycle())
    assert states == (False, True, True, True, True, False, False, False)


def test_constant_root_fit_recovers_cadence_and_world_lock() -> None:
    positions = _paw_cycle()
    states = infer_height_backward_stance(positions)
    fit = fit_constant_root_step(
        [positions],
        [states],
        root_rotation_xyzw=(0.0, 0.0, 0.0, 1.0),
        minimum_step_m=0.005,
        maximum_step_m=0.04,
        grid_step_m=0.0001,
    )
    assert fit.step_m == pytest.approx(0.02, abs=1.0e-12)
    assert fit.maximum_contact_horizontal_step_m == pytest.approx(0.0, abs=1.0e-12)
    assert fit.contact_pair_count == 3


def test_uniform_skin_scale_is_measured_from_corresponding_bones() -> None:
    audit = infer_uniform_skin_linear_scale(
        _skin_document(1.0),
        _skin_document(1.18),
    )
    assert audit.linear_scale == pytest.approx(1.18)
    assert audit.measured_bone_count == 4
    assert audit.maximum_relative_ratio_error == pytest.approx(0.0)


def test_nonuniform_skin_cannot_claim_a_larger_contact_gate() -> None:
    with pytest.raises(WorldContactError, match="not a uniform-scale derivative"):
        infer_uniform_skin_linear_scale(
            _skin_document(1.0),
            _skin_document(1.18, mismatched_last_bone=True),
        )


@pytest.mark.parametrize("scale", (0.82, 1.18))
def test_root_fit_is_dimensionally_equivariant(scale: float) -> None:
    positions = _paw_cycle()
    states = infer_height_backward_stance(positions)
    reference = fit_constant_root_step(
        [positions],
        [states],
        root_rotation_xyzw=(0.0, 0.0, 0.0, 1.0),
    )
    scaled = fit_constant_root_step(
        [positions * scale],
        [states],
        root_rotation_xyzw=(0.0, 0.0, 0.0, 1.0),
        minimum_step_m=0.005 * scale,
        maximum_step_m=0.04 * scale,
        grid_step_m=0.0001 * scale,
    )
    assert scaled.step_m / scale == pytest.approx(reference.step_m, abs=1.0e-12)
    assert scaled.maximum_contact_horizontal_step_m / scale == pytest.approx(
        reference.maximum_contact_horizontal_step_m,
        abs=1.0e-12,
    )


def test_moving_idle_anchor_cannot_claim_high_confidence_contact() -> None:
    idle = np.zeros((4, 4, 3), dtype=np.float64)
    idle[1, 2, 0] = 0.004
    result = evaluate_idle_contact_gate(
        idle,
        maximum_vertical_range_m=0.015,
        maximum_step_displacement_m=0.003,
    )
    assert result["paw_hind_left"]["passed"] is False
    assert result["paw_front_left"]["passed"] is True


def test_low_excursion_walk_cannot_pass_as_dynamic_motion() -> None:
    walk = np.zeros((8, 4, 3), dtype=np.float64)
    walk[:, :, 1] = np.linspace(0.0, 0.001, 8)[:, None]
    result = evaluate_walk_dynamic_gate(
        walk,
        minimum_vertical_range_m=0.005,
    )
    assert all(record["passed"] is False for record in result.values())


@pytest.mark.parametrize(
    "positions",
    [
        np.zeros((2, 3), dtype=np.float64),
        np.full((4, 3), np.nan, dtype=np.float64),
        np.zeros((4, 2), dtype=np.float64),
    ],
)
def test_stance_rejects_malformed_trajectories(positions: np.ndarray) -> None:
    with pytest.raises(WorldContactError):
        infer_height_backward_stance(positions)


def test_stance_rejects_cycles_without_supported_pairs() -> None:
    positions = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [2.0, 0.0, 0.0],
            [3.0, 1.0, 0.0],
            [4.0, 0.0, 0.0],
            [5.0, 1.0, 0.0],
        ],
        dtype=np.float64,
    )
    with pytest.raises(WorldContactError):
        infer_height_backward_stance(positions)

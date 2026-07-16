from __future__ import annotations

import numpy as np
import pytest

from avengine.m2.world_contact import (
    WorldContactError,
    fit_constant_root_step,
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

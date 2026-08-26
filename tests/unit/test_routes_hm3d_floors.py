"""Floor detection has to survive a staircase."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
TOOL_ROOT = REPOSITORY_ROOT / "tools/routes"
if str(TOOL_ROOT) not in sys.path:
    sys.path.insert(0, str(TOOL_ROOT))

from compile_hm3d_dynamic_source_bank import find_floors  # noqa: E402


class ScriptedPathfinder:
    """Returns navigable heights from a fixed list, in order, cycling."""

    def __init__(self, heights):
        self._heights = list(heights)
        self._cursor = 0

    def get_random_navigable_point(self):
        value = self._heights[self._cursor % len(self._heights)]
        self._cursor += 1
        return np.array([0.0, value, 0.0], dtype=np.float64)


def _two_storeys_with_stairs(lower=0.16, upper=3.16, treads=14):
    """Two dense floors joined by a sparse, continuous run of stair treads."""

    heights = [lower] * 600 + [upper] * 300
    step = (upper - lower) / (treads + 1)
    for tread in range(1, treads + 1):
        heights.extend([lower + tread * step] * 3)
    return heights


def test_a_staircase_does_not_merge_two_storeys():
    """The regression this detector exists for.

    Splitting sorted heights on vertical gaps finds no gap to split on, because
    the treads bridge the two floors. Both merge into one cluster whose median
    lands on the busier floor and the other storey is dropped in silence.
    """

    pathfinder = ScriptedPathfinder(_two_storeys_with_stairs())
    floors = find_floors(pathfinder, samples=942, bin_m=0.25, minimum_share=0.05)

    assert len(floors) == 2
    assert floors[0]["height_m"] == pytest.approx(0.16, abs=0.05)
    assert floors[1]["height_m"] == pytest.approx(3.16, abs=0.05)
    # Neither storey may be swallowed by the other.
    assert floors[0]["navigable_share"] > 0.5
    assert floors[1]["navigable_share"] > 0.2


def test_a_single_storey_is_one_floor():
    pathfinder = ScriptedPathfinder([0.12] * 500)
    floors = find_floors(pathfinder, samples=500, bin_m=0.25, minimum_share=0.05)
    assert len(floors) == 1
    assert floors[0]["navigable_share"] == pytest.approx(1.0)


def test_a_floor_straddling_a_bin_boundary_stays_one_floor():
    """A real floor surface is not perfectly flat and must not split in two."""

    heights = list(np.linspace(0.0, 0.30, 400))
    floors = find_floors(
        ScriptedPathfinder(heights), samples=400, bin_m=0.25, minimum_share=0.05
    )
    assert len(floors) == 1


def test_sparse_landings_are_not_reported_as_floors():
    heights = [0.1] * 900 + [1.4] * 20 + [1.9] * 20
    floors = find_floors(
        ScriptedPathfinder(heights), samples=940, bin_m=0.25, minimum_share=0.05
    )
    assert len(floors) == 1
    assert floors[0]["height_m"] == pytest.approx(0.1, abs=0.05)


def test_no_navigable_points_is_no_floors():
    class Empty:
        def get_random_navigable_point(self):
            return np.array([np.nan, np.nan, np.nan])

    assert find_floors(Empty(), samples=10, bin_m=0.25, minimum_share=0.05) == []

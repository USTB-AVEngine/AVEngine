from __future__ import annotations

import numpy as np

from avengine.m5.visual import _topdown_panels


def test_topdown_panel_tracks_both_actors_and_sources() -> None:
    frame_count = 75
    actors = np.zeros((frame_count, 2, 3), dtype=np.float64)
    sources = np.zeros_like(actors)
    actors[:, 0, 0] = np.linspace(0.0, 1.0, frame_count)
    actors[:, 0, 2] = 0.7
    actors[:, 1, 0] = np.linspace(0.25, 1.25, frame_count)
    actors[:, 1, 2] = -0.7
    sources[:] = actors
    sources[:, :, 1] = 0.5
    panels = _topdown_panels(
        navmesh=None,
        navmesh_bounds=None,
        actor_positions=actors,
        source_positions=sources,
        listener_position=(-2.5, 1.55, 0.0),
    )
    assert panels.shape == (75, 240, 240, 3)
    assert panels.dtype == np.uint8
    assert not np.array_equal(panels[0], panels[-1])

from __future__ import annotations

import numpy as np

from avengine.timeline.visual import _topdown_panels


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
        listener_positions=np.repeat(
            np.asarray([[-2.5, 1.55, 0.0]], dtype=np.float64),
            frame_count,
            axis=0,
        ),
    )
    assert panels.shape == (75, 240, 240, 3)
    assert panels.dtype == np.uint8
    assert not np.array_equal(panels[0], panels[-1])


def test_topdown_panel_tracks_a_moving_listener() -> None:
    frame_count = 75
    actors = np.zeros((frame_count, 2, 3), dtype=np.float64)
    sources = np.zeros_like(actors)
    listeners = np.zeros((frame_count, 3), dtype=np.float64)
    listeners[:, 0] = np.linspace(-1.0, 1.0, frame_count)
    listeners[:, 1] = 1.55
    panels = _topdown_panels(
        navmesh=None,
        navmesh_bounds=None,
        actor_positions=actors,
        source_positions=sources,
        listener_positions=listeners,
    )
    assert panels.shape == (75, 240, 240, 3)
    assert not np.array_equal(panels[0], panels[-1])


def test_topdown_panel_accepts_source_labels_and_listener_orientation() -> None:
    frame_count = 75
    sources = np.zeros((frame_count, 1, 3), dtype=np.float64)
    listeners = np.repeat(
        np.asarray([[-1.0, 1.55, 0.0]], dtype=np.float64),
        frame_count,
        axis=0,
    )
    orientations = np.zeros((frame_count, 4), dtype=np.float64)
    orientations[:, 0] = 1.0

    panels = _topdown_panels(
        navmesh=None,
        navmesh_bounds=None,
        actor_positions=sources,
        source_positions=sources,
        listener_positions=listeners,
        listener_orientations_wxyz=orientations,
        actor_labels=("Source 0",),
    )

    assert panels.shape == (75, 240, 240, 3)
    assert panels.dtype == np.uint8

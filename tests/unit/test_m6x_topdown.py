from __future__ import annotations

import numpy as np
import pytest
from PIL import ImageDraw

from avengine.m6x.geometry import RuntimeObstacleMap
from avengine.m6x.topdown import (
    M6XTopdownError,
    render_runtime_topdown_frame,
    render_runtime_topdown_frames,
)


def _obstacle_map() -> RuntimeObstacleMap:
    navmesh = np.ones((40, 40), dtype=np.uint8)
    navmesh[:, 18:22] = 0
    return RuntimeObstacleMap(
        binary_navmesh=navmesh,
        bounds_m=((0.0, 0.0, 0.0), (4.0, 2.0, 4.0)),
        floor_height_m=0.4,
        meters_per_pixel=0.1,
        rigid_obstacles=(
            {
                "object_id": 17,
                "handle": "chair_17",
                "footprint_xz_m": [
                    [2.6, 2.5],
                    [3.4, 2.5],
                    [3.4, 3.2],
                    [2.6, 3.2],
                ],
            },
        ),
    )


def _paths() -> dict[str, list[list[float]]]:
    return {
        "source_human": [
            [0.5, 1.5, 0.5],
            [0.8, 1.5, 0.8],
            [1.1, 1.5, 1.1],
        ],
        "source_dog": [
            [3.5, 0.7, 0.7],
            [3.2, 0.7, 0.9],
            [2.9, 0.7, 1.1],
        ],
    }


def test_draws_navmesh_rigid_obb_listener_fov_and_multiple_source_centers() -> None:
    frame = render_runtime_topdown_frame(
        _obstacle_map(),
        _paths(),
        1,
        listener_position_m=(1.0, 1.47, 3.1),
        listener_yaw_deg=55.0,
        camera_hfov_degrees=105.0,
        source_activity_by_frame={
            "source_human": [False, True, True],
            "source_dog": [True, False, True],
        },
        source_labels={"source_human": "HUMAN", "source_dog": "BEAGLE"},
        source_colors={
            "source_human": (42, 210, 220),
            "source_dog": (250, 120, 70),
        },
        size_wh=(480, 360),
    )
    assert frame.shape == (360, 480, 3)
    assert frame.dtype == np.uint8
    assert frame.flags.c_contiguous

    # Every major authority has a distinct color family in the rendered
    # pixels: dark non-navmesh, orange OBB, blue HFOV, yellow listener and
    # cyan/orange source markers.  This catches regressions that silently
    # drop ReplicaCAD-style rigid furniture from the panel.
    pixels = frame.reshape(-1, 3).astype(np.int16)

    def near(color: tuple[int, int, int], tolerance: int = 8) -> bool:
        return bool(np.any(np.max(np.abs(pixels - color), axis=1) <= tolerance))

    assert near((43, 50, 59))
    assert near((255, 186, 90))
    assert near((46, 154, 255), tolerance=20)
    assert near((255, 224, 66))
    assert near((42, 210, 220))
    assert near((250, 120, 70), tolerance=20)


def test_renders_all_frames_and_current_source_points_move() -> None:
    frames = render_runtime_topdown_frames(
        _obstacle_map(),
        _paths(),
        listener_position_m=(1.0, 1.47, 3.1),
        listener_yaw_deg=55.0,
        camera_hfov_degrees=105.0,
        size_wh=(320, 240),
        rigid_label_limit=0,
    )
    assert frames.shape == (3, 240, 320, 3)
    assert frames.dtype == np.uint8
    assert frames.flags.c_contiguous
    assert not np.array_equal(frames[0], frames[1])
    assert not np.array_equal(frames[1], frames[2])


def test_rejects_ambiguous_frames_activity_and_obstacle_footprints() -> None:
    with pytest.raises(M6XTopdownError, match="frame counts differ"):
        render_runtime_topdown_frames(
            _obstacle_map(),
            {"a": [[0.0, 1.0, 0.0]], "b": [[0.0, 1.0, 0.0]] * 2},
            listener_position_m=(1.0, 1.0, 1.0),
            listener_yaw_deg=0.0,
            camera_hfov_degrees=90.0,
        )

    with pytest.raises(M6XTopdownError, match="activity"):
        render_runtime_topdown_frames(
            _obstacle_map(),
            _paths(),
            listener_position_m=(1.0, 1.0, 1.0),
            listener_yaw_deg=0.0,
            camera_hfov_degrees=90.0,
            source_activity_by_frame={
                "source_human": [True, False, True],
                "source_dog": [1, 0, 1],
            },
        )

    with pytest.raises(M6XTopdownError, match="source_colors"):
        render_runtime_topdown_frames(
            _obstacle_map(),
            _paths(),
            listener_position_m=(1.0, 1.0, 1.0),
            listener_yaw_deg=0.0,
            camera_hfov_degrees=90.0,
            source_colors={"source_human": (42, 210, 220)},
        )

    invalid = RuntimeObstacleMap(
        binary_navmesh=np.ones((10, 10), dtype=np.uint8),
        bounds_m=((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
        floor_height_m=0.0,
        meters_per_pixel=0.1,
        rigid_obstacles=({"footprint_xz_m": [[0.0, 0.0], [0.1, 0.1]]},),
    )
    with pytest.raises(M6XTopdownError, match="footprint"):
        render_runtime_topdown_frames(
            invalid,
            {"source0": [[0.5, 0.5, 0.5]]},
            listener_position_m=(0.5, 0.5, 0.5),
            listener_yaw_deg=0.0,
            camera_hfov_degrees=90.0,
        )


def test_explicitly_draws_centers_only_not_body_volumes() -> None:
    frame = render_runtime_topdown_frame(
        _obstacle_map(),
        _paths(),
        0,
        listener_position_m=(1.0, 1.47, 3.1),
        listener_yaw_deg=55.0,
        camera_hfov_degrees=105.0,
        size_wh=(480, 360),
    )
    # The contract is made visible in the panel itself.  Rendering accepts no
    # body radius/capsule input, so callers cannot accidentally imply it.
    assert frame.shape == (360, 480, 3)
    with pytest.raises(TypeError, match="body_radius"):
        render_runtime_topdown_frame(
            _obstacle_map(),
            _paths(),
            0,
            listener_position_m=(1.0, 1.47, 3.1),
            listener_yaw_deg=55.0,
            camera_hfov_degrees=105.0,
            body_radius_m=0.2,
        )


def test_draws_every_replicacad_style_rigid_footprint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rigid_obstacles = tuple(
        {
            "object_id": object_id,
            "handle": f"replicacad_furniture_{object_id:03d}",
            "footprint_xz_m": [
                [0.05 + object_id * 0.001, 2.5],
                [0.08 + object_id * 0.001, 2.5],
                [0.08 + object_id * 0.001, 2.53],
                [0.05 + object_id * 0.001, 2.53],
            ],
        }
        for object_id in range(113)
    )
    obstacle_map = RuntimeObstacleMap(
        binary_navmesh=np.ones((40, 40), dtype=np.uint8),
        bounds_m=((0.0, 0.0, 0.0), (4.0, 2.0, 4.0)),
        floor_height_m=0.4,
        meters_per_pixel=0.1,
        rigid_obstacles=rigid_obstacles,
    )
    original_polygon = ImageDraw.ImageDraw.polygon
    polygon_calls: list[object] = []

    def recording_polygon(self, xy, *args, **kwargs):
        polygon_calls.append(xy)
        return original_polygon(self, xy, *args, **kwargs)

    monkeypatch.setattr(ImageDraw.ImageDraw, "polygon", recording_polygon)
    render_runtime_topdown_frame(
        obstacle_map,
        {"source0": [[1.0, 1.0, 1.0]]},
        0,
        listener_position_m=(2.0, 1.47, 2.0),
        listener_yaw_deg=0.0,
        camera_hfov_degrees=90.0,
        size_wh=(480, 360),
        rigid_label_limit=0,
    )
    # One polygon per retained rigid footprint, plus the visual-only HFOV
    # wedge.  No high-object-count shortcut may silently omit furniture.
    assert len(polygon_calls) == 114

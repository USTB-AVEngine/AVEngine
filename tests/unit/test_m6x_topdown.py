from __future__ import annotations

import numpy as np
import pytest
from PIL import ImageDraw

from avengine.m6x.geometry import RuntimeObstacleMap
from avengine.m6x.topdown import (
    M6XTopdownError,
    _interior_navmesh_exclusions,
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
                "obstacle_role": "ground_blocker",
                "footprint_xz_m": [
                    [2.6, 2.5],
                    [3.4, 2.5],
                    [3.4, 3.2],
                    [2.6, 3.2],
                ],
            },
            {
                "object_id": 18,
                "handle": "rug_18",
                "obstacle_role": "walkable_floor_covering",
                "footprint_xz_m": [
                    [0.2, 2.4],
                    [0.8, 2.4],
                    [0.8, 2.8],
                    [0.2, 2.8],
                ],
            },
            {
                "object_id": 19,
                "handle": "picture_19",
                "obstacle_role": "elevated_object",
                "footprint_xz_m": [
                    [3.2, 0.2],
                    [3.7, 0.2],
                    [3.7, 0.5],
                    [3.2, 0.5],
                ],
            },
            {
                "object_id": 20,
                "handle": "unresolved_20",
                "obstacle_role": "unknown",
                "footprint_xz_m": [
                    [2.5, 0.3],
                    [2.9, 0.3],
                    [2.9, 0.7],
                    [2.5, 0.7],
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


def test_baked_navmesh_holes_are_separate_from_room_exterior() -> None:
    navmesh = np.ones((12, 12), dtype=np.uint8)
    navmesh[0, :] = 0
    navmesh[-1, :] = 0
    navmesh[:, 0] = 0
    navmesh[:, -1] = 0
    navmesh[1:6, 3] = 0
    navmesh[7:10, 7:10] = 0

    interior, component_count = _interior_navmesh_exclusions(navmesh)

    assert component_count == 1
    assert int(np.count_nonzero(interior)) == 9
    assert np.all(interior[7:10, 7:10])
    assert not np.any(interior[1:6, 3])
    assert not np.any(interior[[0, -1], :])


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
    # pixels: dark non-navmesh; orange blocking, teal walkable, blue elevated
    # and red unresolved OBBs; blue HFOV; yellow listener; and source markers.
    pixels = frame.reshape(-1, 3).astype(np.int16)

    def near(color: tuple[int, int, int], tolerance: int = 8) -> bool:
        return bool(np.any(np.max(np.abs(pixels - color), axis=1) <= tolerance))

    assert near((43, 50, 59))
    assert near((255, 186, 90))
    assert near((76, 222, 194))
    assert near((151, 184, 218))
    assert near((244, 102, 110))
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


def test_per_frame_listener_pose_moves_topdown_listener_and_fov() -> None:
    source = {"source0": [[3.0, 1.0, 1.0]] * 3}
    frames = render_runtime_topdown_frames(
        _obstacle_map(),
        source,
        listener_position_m=(0.5, 1.47, 0.5),
        listener_yaw_deg=0.0,
        listener_positions_m_by_frame=(
            (0.5, 1.47, 0.5),
            (1.0, 1.47, 1.0),
            (1.5, 1.47, 1.5),
        ),
        listener_yaws_deg_by_frame=(0.0, 45.0, 90.0),
        camera_hfov_degrees=90.0,
        size_wh=(320, 240),
        rigid_label_limit=0,
    )
    assert frames.shape == (3, 240, 320, 3)
    assert not np.array_equal(frames[0], frames[1])
    assert not np.array_equal(frames[1], frames[2])

    with pytest.raises(M6XTopdownError, match=r"finite \[frame,3\]"):
        render_runtime_topdown_frames(
            _obstacle_map(),
            source,
            listener_position_m=(0.5, 1.47, 0.5),
            listener_yaw_deg=0.0,
            listener_positions_m_by_frame=((0.5, 1.47, 0.5),),
            listener_yaws_deg_by_frame=(0.0, 45.0, 90.0),
            camera_hfov_degrees=90.0,
        )


def test_explicit_entity_heading_stays_constant_despite_idle_anchor_wobble(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    line_calls: list[tuple[float, float, float, float]] = []
    original_line = ImageDraw.ImageDraw.line

    def recording_line(self, xy, *args, **kwargs):
        if (
            kwargs.get("width") == 2
            and len(xy) == 4
            and all(isinstance(value, (int, float, np.number)) for value in xy)
        ):
            line_calls.append(tuple(float(value) for value in xy))
        return original_line(self, xy, *args, **kwargs)

    monkeypatch.setattr(ImageDraw.ImageDraw, "line", recording_line)
    render_runtime_topdown_frames(
        _obstacle_map(),
        {
            "source_idle": [
                [1.000, 1.5, 1.000],
                [1.006, 1.5, 0.997],
                [0.996, 1.5, 1.004],
            ]
        },
        listener_position_m=(2.0, 1.47, 3.0),
        listener_yaw_deg=0.0,
        camera_hfov_degrees=90.0,
        source_heading_xz_by_frame={"source_idle": [[1.0, 0.0]] * 3},
        size_wh=(320, 240),
        rigid_label_limit=0,
    )
    assert len(line_calls) == 3
    directions = []
    for x0, y0, x1, y1 in line_calls:
        direction = np.asarray((x1 - x0, y1 - y0), dtype=np.float64)
        directions.append(direction / np.linalg.norm(direction))
    assert np.allclose(directions, directions[0], rtol=0.0, atol=1.0e-12)

    line_calls.clear()
    render_runtime_topdown_frames(
        _obstacle_map(),
        {
            "source_idle": [
                [1.000, 1.5, 1.000],
                [1.006, 1.5, 0.997],
                [0.996, 1.5, 1.004],
            ]
        },
        listener_position_m=(2.0, 1.47, 3.0),
        listener_yaw_deg=0.0,
        camera_hfov_degrees=90.0,
        size_wh=(320, 240),
        rigid_label_limit=0,
    )
    assert line_calls == []


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

    with pytest.raises(M6XTopdownError, match="headings"):
        render_runtime_topdown_frames(
            _obstacle_map(),
            _paths(),
            listener_position_m=(1.0, 1.0, 1.0),
            listener_yaw_deg=0.0,
            camera_hfov_degrees=90.0,
            source_heading_xz_by_frame={
                "source_human": [[1.0, 0.0]] * 3,
                "source_dog": [[0.0, 0.0]] * 3,
            },
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


def test_elevated_obstacle_fill_is_alpha_composited() -> None:
    obstacle_map = RuntimeObstacleMap(
        binary_navmesh=np.ones((20, 20), dtype=np.uint8),
        bounds_m=((0.0, 0.0, 0.0), (2.0, 2.0, 2.0)),
        floor_height_m=0.0,
        meters_per_pixel=0.1,
        rigid_obstacles=(
            {
                "object_id": 1,
                "handle": "elevated_lamp",
                "obstacle_role": "elevated_object",
                "footprint_xz_m": [
                    [0.7, 0.7],
                    [1.3, 0.7],
                    [1.3, 1.3],
                    [0.7, 1.3],
                ],
            },
        ),
    )
    frame = render_runtime_topdown_frame(
        obstacle_map,
        {"source0": [[1.8, 1.0, 1.8]]},
        0,
        listener_position_m=(0.2, 1.47, 0.2),
        listener_yaw_deg=0.0,
        camera_hfov_degrees=90.0,
        size_wh=(480, 360),
        rigid_label_limit=0,
    )

    # The room center lies inside the elevated footprint.  Its blue alpha-36
    # fill must be blended over the light-gray navmesh, not written as opaque
    # RGB and then stripped of alpha.
    center = frame[183, 239]
    blue = np.asarray((104, 135, 168), dtype=np.uint8)
    navmesh = np.asarray((209, 219, 225), dtype=np.uint8)
    assert np.all(center > blue)
    assert np.all(center < navmesh)

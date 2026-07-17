from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pytest

from avengine.m5_1.legacy_route import FRAME_COUNT
from avengine.m5_1.mp3d_capture import (
    MP3DCaptureError,
    _assert_route_geometry,
    _pathfinder_path_record,
    derive_mp3d_route_paths,
    load_mp3d_route_manifest,
    write_mp3d_contact_sheet,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
ROUTE_PATH = (
    REPOSITORY_ROOT
    / "examples"
    / "m5_1"
    / "mp3d_articulated_review"
    / "route_manifest.json"
)


def test_example_route_is_exact_parallel_separated_motion() -> None:
    route = load_mp3d_route_manifest(ROUTE_PATH)
    paths = derive_mp3d_route_paths(route)
    assert paths.human.shape == (FRAME_COUNT, 3)
    assert paths.beagle.shape == (FRAME_COUNT, 3)
    assert np.array_equal(paths.human[0], np.asarray([-4.6, 0.072447, -2.7]))
    assert np.array_equal(paths.human[-1], np.asarray([-4.6, 0.072447, -3.8]))
    assert np.array_equal(paths.beagle[0], np.asarray([-3.7, 0.072447, -2.7]))
    assert np.array_equal(paths.beagle[-1], np.asarray([-3.7, 0.072447, -3.8]))

    geometry = _assert_route_geometry(route, paths)
    assert geometry["minimum_center_separation_m"] == pytest.approx(0.9)
    assert geometry["maximum_center_separation_m"] == pytest.approx(0.9)
    for actor_id in ("human0", "dog0"):
        assert geometry["movement"][actor_id]["path_length_m"] == pytest.approx(1.1)
        assert geometry["movement"][actor_id][
            "endpoint_displacement_m"
        ] == pytest.approx(1.1)
        assert geometry["movement"][actor_id][
            "maximum_center_step_m"
        ] == pytest.approx(1.1 / 269.0)


def test_route_geometry_rejects_center_overlap() -> None:
    route = load_mp3d_route_manifest(ROUTE_PATH)
    invalid = copy.deepcopy(route)
    invalid["routes"]["dog0"] = copy.deepcopy(invalid["routes"]["human0"])
    paths = derive_mp3d_route_paths(invalid)
    with pytest.raises(MP3DCaptureError, match="minimum separation"):
        _assert_route_geometry(invalid, paths)


class _FakePathFinder:
    def __init__(self, *, failed_frame: int | None = None, step_offset: float = 0.0):
        self.failed_frame = failed_frame
        self.step_offset = step_offset
        self.calls = 0

    def is_navigable(self, point: np.ndarray, maximum_y_delta: float) -> bool:
        del point
        assert maximum_y_delta == pytest.approx(1.0e-4)
        index = self.calls
        self.calls += 1
        return index != self.failed_frame

    @staticmethod
    def snap_point(point: np.ndarray) -> np.ndarray:
        return np.asarray(point, dtype=np.float64)

    @staticmethod
    def get_island(point: np.ndarray) -> int:
        del point
        return 1

    def try_step_no_sliding(
        self, start: np.ndarray, end: np.ndarray
    ) -> np.ndarray:
        del start
        result = np.asarray(end, dtype=np.float64).copy()
        result[0] += self.step_offset
        return result


def test_pathfinder_record_requires_every_point_and_no_sliding_segment() -> None:
    route = load_mp3d_route_manifest(ROUTE_PATH)
    path = derive_mp3d_route_paths(route).human
    record = _pathfinder_path_record(
        _FakePathFinder(),
        path,
        owner="human0",
        maximum_snap_error_m=1.0e-5,
        maximum_y_delta_m=1.0e-4,
        maximum_step_endpoint_error_m=1.0e-5,
    )
    assert record["navigable_frame_count"] == FRAME_COUNT
    assert record["island_id"] == 1
    assert record["no_sliding_passed_segment_count"] == FRAME_COUNT - 1
    assert record["maximum_snap_error_m"] == 0.0
    assert record["maximum_step_endpoint_error_m"] == 0.0

    with pytest.raises(MP3DCaptureError, match="non-navigable center frames"):
        _pathfinder_path_record(
            _FakePathFinder(failed_frame=17),
            path,
            owner="human0",
            maximum_snap_error_m=1.0e-5,
            maximum_y_delta_m=1.0e-4,
            maximum_step_endpoint_error_m=1.0e-5,
        )
    with pytest.raises(MP3DCaptureError, match="no-sliding segment failures"):
        _pathfinder_path_record(
            _FakePathFinder(step_offset=2.0e-5),
            path,
            owner="human0",
            maximum_snap_error_m=1.0e-5,
            maximum_y_delta_m=1.0e-4,
            maximum_step_endpoint_error_m=1.0e-5,
        )


def test_contact_sheet_retains_nine_semantic_box_frames(tmp_path: Path) -> None:
    rgb = np.zeros((FRAME_COUNT, 8, 10, 3), dtype=np.uint8)
    rgb[..., 1] = 48
    semantic = np.zeros((FRAME_COUNT, 8, 10), dtype=np.int32)
    semantic[:, 1:5, 1:4] = 62000
    semantic[:, 4:7, 6:9] = 62001
    destination = tmp_path / "sheet.jpg"
    record = write_mp3d_contact_sheet(
        rgb=rgb,
        semantic=semantic,
        semantic_ids={"human0": 62000, "dog0": 62001},
        output_path=destination,
    )
    assert destination.is_file()
    assert record["readback_verified"] is True
    assert record["file"]["path"] == "sheet.jpg"
    assert record["size_wh"] == [30, 96]
    assert len(record["selected_frames"]) == 9
    assert all(
        item["visible_pixels"] == {"human0": 12, "dog0": 9}
        for item in record["selected_frames"]
    )

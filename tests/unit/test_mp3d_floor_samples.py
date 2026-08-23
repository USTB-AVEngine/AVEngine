from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from avengine.m5.mp3d_floor_samples import (
    MP3DFloorSampleError,
    export_live_mp3d_floor_samples,
)
from tools.m5 import export_mp3d_camera_floor_samples as tool


REPOSITORY = Path(__file__).resolve().parents[2]
ROOM_MANIFEST = REPOSITORY / "examples/m1/rooms/habitat_mp3d_example/room_manifest.json"
BASE_REQUEST = REPOSITORY / "examples/m1/requests/habitat_mp3d_example.json"
ROOM_ID = "habitat_mp3d_example_17DRP5sb8fy"


def _routes() -> dict[str, list[list[float]]]:
    return {
        "actor0": [[0.6, 0.0, 0.6], [1.4, 0.0, 1.4]],
        "actor1": [[0.6, 0.0, 2.6], [1.4, 0.0, 3.4]],
    }


class _PathFinder:
    is_loaded = True

    def __init__(self, *, candidate_snap_offset_x=0.0, invalid_bounds=False):
        self.binary = np.ones((4, 4), dtype=np.uint8)
        self.bounds = np.asarray(
            [[0.0, -1.0, 0.0], [4.0, 2.0, 4.0]], dtype=np.float64
        )
        if invalid_bounds:
            self.bounds[1, 0] = np.nan
        self.route_points = {
            tuple(point) for points in _routes().values() for point in points
        }
        self.candidate_snap_offset_x = candidate_snap_offset_x

    def get_topdown_view(self, meters_per_pixel, floor_height):
        assert float(meters_per_pixel) == pytest.approx(1.0)
        assert float(floor_height) == pytest.approx(0.0)
        return self.binary.copy()

    def get_bounds(self):
        return self.bounds.copy()

    def is_navigable(self, point, maximum_y_delta):
        value = np.asarray(point, dtype=np.float64)
        return bool(np.all(np.isfinite(value)) and maximum_y_delta >= 0.0)

    def snap_point(self, point):
        value = np.asarray(point, dtype=np.float64).copy()
        if tuple(value.tolist()) not in self.route_points:
            value[0] += self.candidate_snap_offset_x
        return value

    def get_island(self, point):
        return 7 if float(np.asarray(point)[0]) < 2.0 else 8

    def distance_to_closest_obstacle(self, _point, _maximum):
        return 1.0

    def load_nav_mesh(self, _path):
        self.is_loaded = True
        return True


def test_live_export_is_sorted_bounded_and_rejects_other_islands() -> None:
    result = export_live_mp3d_floor_samples(
        _PathFinder(),
        room_id=ROOM_ID,
        actor_routes_m=_routes(),
        meters_per_pixel=1.0,
        maximum_sample_count=4,
        minimum_sample_count=4,
        candidate_scan_multiplier=4,
    )

    samples = result["camera_floor_samples_m"]
    assert len(samples) == 4
    assert samples == sorted(samples)
    assert all(point[0] < 2.0 for point in samples)
    assert result["coordinate_space"] == "habitat_world_xyz_m"
    assert result["native_habitat_floor_export"] is True
    assert result["native_habitat_pixel_status"] == "pending"
    evidence = result["machine_evidence"]
    assert evidence["required_actor_route_island_id"] == 7
    assert any(
        row["failure_reason"] == "different_island"
        for row in evidence["sample_readbacks"]
    )
    for row in evidence["sample_readbacks"]:
        assert row["coordinate_space"] == "habitat_world_xyz_m"
        assert row["authority"]
        assert len(row["requested_xyz_m"]) == 3


def test_actor_routes_must_share_one_live_pathfinder_island() -> None:
    routes = _routes()
    routes["actor1"] = [[2.6, 0.0, 2.6], [3.4, 0.0, 3.4]]
    with pytest.raises(MP3DFloorSampleError, match="cross PathFinder islands"):
        export_live_mp3d_floor_samples(
            _PathFinder(),
            room_id=ROOM_ID,
            actor_routes_m=routes,
            meters_per_pixel=1.0,
        )


def test_candidate_snap_error_fails_when_too_few_samples_survive() -> None:
    with pytest.raises(
        MP3DFloorSampleError,
        match="only 0 accepted floor samples.*horizontal_snap_error",
    ):
        export_live_mp3d_floor_samples(
            _PathFinder(candidate_snap_offset_x=0.2),
            room_id=ROOM_ID,
            actor_routes_m=_routes(),
            meters_per_pixel=1.0,
            maximum_sample_count=4,
            minimum_sample_count=4,
            maximum_horizontal_snap_error_m=0.1,
        )


def test_nonfinite_pathfinder_bounds_are_rejected() -> None:
    with pytest.raises(MP3DFloorSampleError, match="bounds must be finite"):
        export_live_mp3d_floor_samples(
            _PathFinder(invalid_bounds=True),
            room_id=ROOM_ID,
            actor_routes_m=_routes(),
            meters_per_pixel=1.0,
        )


def _tool_inputs(tmp_path: Path) -> tuple[argparse.Namespace, _PathFinder]:
    route = tmp_path / "route.json"
    route.write_text(
        json.dumps(
            {
                "research_only": True,
                "episode_counted": False,
                "provenance": {"new_request_room_id": ROOM_ID},
                "native_pathfinder": {"all_paths_share_single_island": True},
                "current_visual_realization": {"actor_skin_root_paths_m": _routes()},
            }
        ),
        encoding="utf-8",
    )
    runtime = tmp_path / "runtime"
    magnum = tmp_path / "magnum"
    rlr = tmp_path / "rlr"
    mp3d = tmp_path / "mp3d"
    pbr = tmp_path / "pbr"
    for directory in (runtime, magnum, rlr, mp3d, pbr):
        directory.mkdir()
    (mp3d / "scene_datasets").mkdir()
    args = tool.parse_args(
        [
            "--room-manifest", str(ROOM_MANIFEST),
            "--base-m1-request", str(BASE_REQUEST),
            "--route-explanation", str(route),
            "--runtime-prefix", str(runtime),
            "--magnum-site", str(magnum),
            "--rlr-sdk-root", str(rlr),
            "--mp3d-root", str(mp3d),
            "--pbr-root", str(pbr),
            "--output", str(tmp_path / "output"),
        ]
    )
    return args, _PathFinder()


def _patch_tool_runtime(monkeypatch, tmp_path: Path, pathfinder: _PathFinder) -> None:
    navmesh = tmp_path / "scene.navmesh"
    navmesh.touch()

    class _Simulator:
        def __init__(self, _configuration):
            self.pathfinder = pathfinder

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

    fake_habitat = SimpleNamespace(Simulator=_Simulator)
    fake_runtime = SimpleNamespace(
        prefix=tmp_path / "runtime",
        mp3d_root=tmp_path / "mp3d",
        pbr_asset_root=tmp_path / "pbr",
        magnum_python_site=tmp_path / "magnum",
        physics_config_path=tmp_path / "physics.json",
        habitat_sim=fake_habitat,
    )
    monkeypatch.setattr(tool, "prepare_installed_habitat_runtime", lambda **_kw: fake_runtime)
    monkeypatch.setattr(tool, "validate_scene_asset_graph", lambda *_a, **_kw: [])
    monkeypatch.setattr(
        tool,
        "_make_configuration",
        lambda *_a, **_kw: (object(), {}, "listener0", {"navmesh": navmesh}),
    )
    monkeypatch.setattr(
        tool,
        "validate_loaded_scene_asset_graph",
        lambda *_a, **_kw: ([], {"status": "pass"}),
    )
    monkeypatch.setattr(
        tool,
        "export_live_mp3d_floor_samples",
        lambda *_a, **_kw: {
            "room_id": ROOM_ID,
            "coordinate_space": "habitat_world_xyz_m",
            "camera_floor_samples_m": [[0.5, 0.0, 0.5]],
            "research_only": True,
            "episode_counted": False,
            "native_habitat_floor_export": True,
        },
    )


def test_tool_serialization_failure_cleans_exact_staging(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args, pathfinder = _tool_inputs(tmp_path)
    _patch_tool_runtime(monkeypatch, tmp_path, pathfinder)
    original_write = tool.write_json

    def failing_write(path, value):
        if Path(path).name == "camera_floor_samples.json":
            raise TypeError("fixture serialization failure")
        original_write(path, value)

    monkeypatch.setattr(tool, "write_json", failing_write)
    with pytest.raises(TypeError, match="fixture serialization failure"):
        tool.run(args)
    assert not args.output.exists()
    assert not list(tmp_path.glob(f".{args.output.name}.staging-*"))


def test_tool_publish_race_preserves_winner_and_cleans_staging(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args, pathfinder = _tool_inputs(tmp_path)
    _patch_tool_runtime(monkeypatch, tmp_path, pathfinder)

    def losing_publish(_policy, _staging, destination):
        Path(destination).mkdir()
        (Path(destination) / "winner.txt").write_text("keep", encoding="utf-8")
        raise FileExistsError("fixture race")

    monkeypatch.setattr(tool, "atomic_publish_directory", losing_publish)
    with pytest.raises(tool.MP3DFloorExportError, match="fixture race"):
        tool.run(args)
    assert (args.output / "winner.txt").read_text(encoding="utf-8") == "keep"
    assert not list(tmp_path.glob(f".{args.output.name}.staging-*"))

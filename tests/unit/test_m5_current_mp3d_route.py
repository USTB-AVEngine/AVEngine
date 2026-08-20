from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from avengine import cli
from avengine.m5 import current_mp3d_route as current_route
from avengine.m5_1.mp3d_capture import _pathfinder_path_record


class _ShortestPath:
    requested_start: np.ndarray
    requested_end: np.ndarray
    points: list[np.ndarray]
    geodesic_distance: float


class _Habitat:
    ShortestPath = _ShortestPath


class _PathFinder:
    is_loaded = True

    def is_navigable(self, point: np.ndarray, maximum_y_delta_m: float) -> bool:
        del maximum_y_delta_m
        return np.asarray(point, dtype=np.float64).shape == (3,)

    def snap_point(self, point: np.ndarray) -> np.ndarray:
        return np.asarray(point, dtype=np.float64).copy()

    def distance_to_closest_obstacle(
        self, point: np.ndarray, maximum_search_radius: float
    ) -> float:
        del point, maximum_search_radius
        return 2.0

    def find_path(self, query: _ShortestPath) -> bool:
        start = np.asarray(query.requested_start, dtype=np.float64)
        end = np.asarray(query.requested_end, dtype=np.float64)
        query.points = [start, end]
        query.geodesic_distance = float(np.linalg.norm(end - start))
        return True

    def get_island(self, point: np.ndarray) -> int:
        del point
        return 4

    def try_step_no_sliding(self, start: np.ndarray, end: np.ndarray) -> np.ndarray:
        del start
        return np.asarray(end, dtype=np.float64).copy()


class _Region:
    def sample_points_m(self) -> np.ndarray:
        return np.asarray(
            [
                [0.0, 0.0, 0.0],
                [1.6, 0.0, 0.0],
                [0.0, 0.0, 1.6],
                [1.6, 0.0, 1.6],
            ],
            dtype=np.float64,
        )


def test_select_base_skin_path_uses_xz_endpoint_coordinates() -> None:
    path = current_route._select_base_skin_path(
        habitat_sim=_Habitat,
        pathfinder=_PathFinder(),
        region=_Region(),
        seed=20_260_820,
        target_distance_m=1.6,
        distance_tolerance_m=0.15,
        maximum_snap_error_m=0.03,
        maximum_y_delta_m=0.25,
        minimum_navmesh_clearance_m=0.10,
        visible_from_current_camera=lambda _point: True,
    )

    assert path.shape == (75, 3)
    assert np.allclose(path[:15], path[0])
    assert np.allclose(path[-15:], path[-1])
    assert float(np.linalg.norm(path[59, (0, 2)] - path[15, (0, 2)])) == pytest.approx(
        1.6
    )


class _BaseClearancePathFinder(_PathFinder):
    def distance_to_closest_obstacle(
        self, point: np.ndarray, maximum_search_radius: float
    ) -> float:
        del maximum_search_radius
        return 0.05 if np.allclose(point, [0.0, 0.0, 0.0]) else 2.0


def test_select_base_skin_path_honors_a_rejected_selection_predicate() -> None:
    with pytest.raises(
        current_route.CurrentMP3DRouteError,
        match="offset-safe native pathfinder endpoints",
    ):
        current_route._select_base_skin_path(
            habitat_sim=_Habitat,
            pathfinder=_PathFinder(),
            region=_Region(),
            seed=20_260_820,
            target_distance_m=1.6,
            distance_tolerance_m=0.15,
            maximum_snap_error_m=0.03,
            maximum_y_delta_m=0.25,
            minimum_navmesh_clearance_m=0.10,
            visible_from_current_camera=lambda _point: False,
        )


def test_offset_feasibility_also_requires_primary_m2_path_clearance() -> None:
    assert not current_route._point_has_offset_clearance(
        _BaseClearancePathFinder(),
        np.asarray([0.0, 0.0, 0.0]),
        maximum_snap_error_m=0.03,
        maximum_y_delta_m=0.25,
        minimum_navmesh_clearance_m=0.10,
        visible_from_current_camera=lambda _point: True,
    )


def test_research_camera_candidates_score_xz_without_tuple_indexing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    room = SimpleNamespace(
        room={"navigation": {"agent_height_m": 1.5}},
        request={"request_id": "m1-current"},
    )
    monkeypatch.setattr(
        current_route,
        "apply_camera_listener_pose",
        lambda _request, **kwargs: {
            "request_id": kwargs["request_id"],
            "primary_camera_rig": {
                "world_from_rig": {"translation_m": kwargs["position_m"]}
            },
        },
    )
    monkeypatch.setattr(
        current_route,
        "_camera_frustum_predicate",
        lambda _request: lambda _point: True,
    )
    paths = {
        actor_id: np.repeat(
            np.asarray([[0.0, 0.0, -3.0]], dtype=np.float64),
            75,
            axis=0,
        )
        for actor_id in current_route.CURRENT_ACTOR_IDS
    }

    candidates = current_route._research_camera_candidates(
        room_inputs=room,
        region=_Region(),
        pathfinder=_PathFinder(),
        visual_paths=paths,
        required_island_id=4,
        seed=20_260_820,
        maximum_snap_error_m=0.03,
        maximum_y_delta_m=0.25,
        minimum_navmesh_clearance_m=0.10,
    )

    assert candidates
    assert candidates[0][1]["camera_island_id"] == 4
    assert candidates[0][1]["camera_floor_navigation"]["navigable"] is True


def test_research_camera_candidates_reject_directly_non_navigable_floor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class UnnavigablePathFinder(_PathFinder):
        def is_navigable(self, point: np.ndarray, maximum_y_delta_m: float) -> bool:
            del point, maximum_y_delta_m
            return False

    room = SimpleNamespace(
        room={"navigation": {"agent_height_m": 1.5}},
        request={"request_id": "m1-current"},
    )
    monkeypatch.setattr(
        current_route,
        "apply_camera_listener_pose",
        lambda _request, **kwargs: {"request_id": kwargs["request_id"]},
    )
    monkeypatch.setattr(
        current_route,
        "_camera_frustum_predicate",
        lambda _request: lambda _point: True,
    )
    paths = {
        actor_id: np.repeat(
            np.asarray([[0.0, 0.0, -3.0]], dtype=np.float64),
            75,
            axis=0,
        )
        for actor_id in current_route.CURRENT_ACTOR_IDS
    }

    with pytest.raises(current_route.CurrentMP3DRouteError, match="no same-island"):
        current_route._research_camera_candidates(
            room_inputs=room,
            region=_Region(),
            pathfinder=UnnavigablePathFinder(),
            visual_paths=paths,
            required_island_id=4,
            seed=20_260_820,
            maximum_snap_error_m=0.03,
            maximum_y_delta_m=0.25,
            minimum_navmesh_clearance_m=0.10,
        )


def test_camera_yaw_toward_uses_xz_projection() -> None:
    assert current_route._camera_yaw_toward(
        np.asarray([0.0, 1.5, 0.0]),
        np.asarray([0.0, 0.3, -4.0]),
    ) == pytest.approx(0.0)
    assert current_route._camera_yaw_toward(
        np.asarray([0.0, 1.5, 0.0]),
        np.asarray([4.0, 0.3, 0.0]),
    ) == pytest.approx(-90.0)


def test_current_camera_frustum_prefilter_rejects_behind_and_edge_points() -> None:
    request = {
        "primary_camera_rig": {
            "world_from_rig": {
                "translation_m": [0.0, 1.5, 0.0],
                "rotation_xyzw": [0.0, 0.0, 0.0, 1.0],
            },
            "shared_calibration": {
                "resolution_hw": [240, 320],
                "hfov_degrees": 90.0,
                "near_m": 0.05,
                "far_m": 100.0,
            },
        }
    }
    visible = current_route._camera_frustum_predicate(request)

    assert visible(np.asarray([0.0, 0.0, -3.0]))
    assert not visible(np.asarray([0.0, 0.0, 3.0]))
    assert not visible(np.asarray([4.0, 0.0, -3.0]))


def test_shared_pathfinder_validator_accepts_75_without_new_digest() -> None:
    path = np.stack(
        (
            np.linspace(0.0, 1.6, 75),
            np.zeros(75),
            np.zeros(75),
        ),
        axis=1,
    )
    record = _pathfinder_path_record(
        _PathFinder(),
        path,
        owner="current route",
        maximum_snap_error_m=0.03,
        maximum_y_delta_m=0.25,
        maximum_step_endpoint_error_m=0.03,
        expected_frame_count=75,
        include_trajectory_sha256=False,
    )

    assert record["frame_count"] == 75
    assert record["segment_count"] == 74
    assert record["unique_island_count"] == 1
    assert "trajectory_sha256" not in record


def test_cli_exposes_explicit_current_mp3d_two_beagle_author(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    calls: dict[str, object] = {}

    def fake_author(**kwargs: object) -> dict[str, object]:
        calls.update(kwargs)
        return {
            "status": "research_only",
            "research_only": True,
            "episode_counted": False,
            "output_directory": "/data/avengine_external/review/fresh",
            "primary_m2_request": "/data/avengine_external/review/fresh/primary_m2_request.json",
            "research_m1_request": "/data/avengine_external/review/fresh/research_m1_request.json",
            "explanation": "/data/avengine_external/review/fresh/two_beagle_route_explanation.json",
            "frame_count": 75,
            "actor_ids": ["actor0", "actor1"],
        }

    monkeypatch.setattr(cli, "author_current_mp3d_two_beagle_route", fake_author)
    arguments = [
        "m5",
        "author-current-mp3d-two-beagle-route",
        "--source-animal-manifest",
        "asset.json",
        "--source-m2-request",
        "source.json",
        "--runtime-prefix",
        "prefix",
        "--mp3d-root",
        "mp3d",
        "--magnum-python-site",
        "magnum",
        "--output",
        str(tmp_path / "fresh"),
    ]
    parsed = cli.build_parser().parse_args(arguments)
    assert parsed.m5_command == "author-current-mp3d-two-beagle-route"
    assert parsed.seed == 20_260_820
    assert parsed.distance_tolerance_m == pytest.approx(0.15)
    assert parsed.minimum_center_separation_m == pytest.approx(0.75)
    assert cli.main(arguments) == 0
    assert calls == {
        "source_animal_manifest_path": "asset.json",
        "source_m2_request_path": "source.json",
        "runtime_prefix": "prefix",
        "mp3d_root": "mp3d",
        "magnum_python_site": "magnum",
        "output_directory": str(tmp_path / "fresh"),
        "seed": 20_260_820,
        "distance_tolerance_m": 0.15,
        "minimum_center_separation_m": 0.75,
    }
    result = __import__("json").loads(capsys.readouterr().out)
    assert result["status"] == "research_only"
    assert result["research_m1_request"].endswith("research_m1_request.json")


def test_route_output_must_be_fresh_immediate_external_child(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(current_route, "EXTERNAL_REVIEW_ROOT", tmp_path)
    output = tmp_path / "fresh"
    assert current_route._fresh_external_output(output) == output

    output.mkdir()
    with pytest.raises(
        current_route.CurrentMP3DRouteError, match="refusing to replace"
    ):
        current_route._fresh_external_output(output)
    with pytest.raises(
        current_route.CurrentMP3DRouteError, match="immediate fresh child"
    ):
        current_route._fresh_external_output(tmp_path / "nested" / "route")

from __future__ import annotations

from copy import deepcopy
import math
from types import SimpleNamespace

import numpy as np
import pytest

from avengine.m5_1.camera_candidate_gate import (
    CameraCandidateGateError,
    HabitatRuntimeCameraProvider,
    evaluate_camera_candidates,
)


class FakePathFinder:
    def __init__(self) -> None:
        self.listener_navigable = True
        self.snap_offset = np.zeros(3, dtype=np.float64)
        self.clearance = 1.0
        self.listener_island = 7
        self.actor_island = 7
        self.bad_actor_x: float | None = None
        self.is_loaded = True

    def is_navigable(self, point: np.ndarray, maximum_y_delta: float) -> bool:
        assert maximum_y_delta == 0.2
        if float(point[0]) == 0.0 and float(point[2]) == 0.0:
            return self.listener_navigable
        return self.bad_actor_x is None or float(point[0]) != self.bad_actor_x

    def snap_point(self, point: np.ndarray) -> np.ndarray:
        if float(point[0]) == 0.0 and float(point[2]) == 0.0:
            return np.asarray(point, dtype=np.float64) + self.snap_offset
        return np.asarray(point, dtype=np.float64)

    def distance_to_closest_obstacle(
        self, point: np.ndarray, maximum_distance: float
    ) -> float:
        assert point.shape == (3,)
        assert maximum_distance == 10.0
        return self.clearance

    def get_island(self, point: np.ndarray) -> int:
        if float(point[0]) == 0.0 and float(point[2]) == 0.0:
            return self.listener_island
        return self.actor_island


class FakeRuntimeProvider:
    def __init__(self) -> None:
        self.pathfinder = FakePathFinder()
        self.runtime_context = {
            "provider_id": "mp3d-physics-simulator-01",
            "scene_id": "17DRP5sb8fy",
            "pathfinder_loaded": True,
            "physics_enabled": True,
            "raycast_enabled": True,
            "room_bounds_source": "loaded_scene_pathfinder_bounds",
        }
        self.nearest_hit = None
        self.room_bounds_m = {
            "minimum_m": [-3.0, -0.1, -4.0],
            "maximum_m": [3.0, 3.0, 1.0],
        }

    def line_of_sight_nearest_hit(
        self, origin: list[float], target: list[float]
    ) -> dict[str, object]:
        nearest = (
            self.nearest_hit(origin, target) if callable(self.nearest_hit) else None
        )
        return {
            "provider_id": self.runtime_context["provider_id"],
            "scene_id": self.runtime_context["scene_id"],
            "physics_enabled": self.runtime_context["physics_enabled"],
            "raycast_enabled": self.runtime_context["raycast_enabled"],
            "endpoint_policy": "full_ray_buffer_zero_no_endpoint_tolerance",
            "nearest_hit_distance_m": nearest,
        }


def _inputs() -> dict[str, object]:
    floors = {
        "source1_actor": [[-1.0, 0.0, -2.0], [-1.0, 0.0, -2.0]],
        "source2_actor": [[1.0, 0.0, -2.0], [1.0, 0.0, -2.0]],
    }
    anchors = {
        actor_id: {
            "torso": [[point[0], 1.1, point[2]] for point in path],
            "mouth": [[point[0], 1.6, point[2]] for point in path],
        }
        for actor_id, path in floors.items()
    }
    return {
        "runtime_provider": FakeRuntimeProvider(),
        "candidates": [
            {
                "candidate_id": "camera_b",
                "priority": 2,
                "position_m": [0.0, 1.5, 0.0],
                "yaw_deg": 0.0,
            },
            {
                "candidate_id": "camera_a",
                "priority": 1,
                "position_m": [0.0, 1.5, 0.0],
                "yaw_deg": 0.0,
            },
        ],
        "actor_floor_paths_m": floors,
        "actor_visibility_anchors_m": anchors,
        "floor_height_m": 0.0,
        "evaluation_id": "mp3d/runtime-camera-gate-v1",
        "maximum_y_delta_m": 0.2,
        "maximum_snap_error_m": 0.05,
        "minimum_clearance_m": 0.5,
        "line_of_sight_tolerance_m": 0.03,
    }


def _evaluate(inputs: dict[str, object]) -> list[dict[str, object]]:
    return evaluate_camera_candidates(**inputs)


def test_passing_candidates_are_stable_and_emit_consumable_room_gates() -> None:
    result = _evaluate(_inputs())

    assert [record["candidate_id"] for record in result] == ["camera_a", "camera_b"]
    for record in result:
        assert record["status"] == "pass"
        room_gate = record["room_gate"]
        assert room_gate["status"] == "pass"
        assert room_gate["provenance"] == "habitat_cpu_runtime"
        assert room_gate["native_habitat_validation_status"] == "pass"
        assert room_gate["line_of_sight_validation_status"] == "pass"
        assert room_gate["full_body_clearance_status"] == "pending_live_ue"
        assert all(
            gate["status"] == "pass" for gate in room_gate["hard_gates"].values()
        )
        assert record["evidence"]["claim_boundary"] == {
            "listener_navmesh_runtime_validated": True,
            "line_of_sight_runtime_queries_complete": True,
            "live_ue_full_body_clearance_validated": False,
            "native_pixel_validation_status": "pending",
            "qualification_claim": False,
            "formal_dataset_count": 0,
        }


@pytest.mark.parametrize(
    ("mutation", "failed_gate"),
    [
        ("listener_navigable", "listener_navmesh"),
        ("listener_snap", "listener_navmesh"),
        ("listener_clearance", "listener_navmesh"),
        ("actor_island", "same_navmesh_island"),
        ("actor_navigable", "same_navmesh_island"),
        ("room_bounds", "room_bounds"),
        ("line_of_sight", "line_of_sight_source1_actor"),
    ],
)
def test_any_failed_runtime_gate_withholds_room_gate(
    mutation: str, failed_gate: str
) -> None:
    inputs = _inputs()
    inputs["candidates"] = [deepcopy(inputs["candidates"][0])]  # type: ignore[index]
    provider = inputs["runtime_provider"]
    pathfinder = provider.pathfinder  # type: ignore[attr-defined]
    if mutation == "listener_navigable":
        pathfinder.listener_navigable = False  # type: ignore[attr-defined]
    elif mutation == "listener_snap":
        pathfinder.snap_offset = np.asarray([0.1, 0.0, 0.0])  # type: ignore[attr-defined]
    elif mutation == "listener_clearance":
        pathfinder.clearance = 0.1  # type: ignore[attr-defined]
    elif mutation == "actor_island":
        pathfinder.actor_island = 9  # type: ignore[attr-defined]
    elif mutation == "actor_navigable":
        pathfinder.bad_actor_x = -1.0  # type: ignore[attr-defined]
    elif mutation == "room_bounds":
        provider.room_bounds_m = {  # type: ignore[attr-defined]
            "minimum_m": [-3.0, -0.1, -4.0],
            "maximum_m": [-0.1, 3.0, 1.0],
        }
    else:
        provider.nearest_hit = lambda _origin, target: (  # type: ignore[attr-defined]
            0.5 if target[0] < 0.0 else None
        )

    record = _evaluate(inputs)[0]
    assert record["status"] == "fail"
    assert record["room_gate"] is None
    assert record["evidence"]["hard_gates"][failed_gate]["status"] == "fail"


def test_line_of_sight_endpoint_tolerance_is_explicit() -> None:
    inputs = _inputs()
    inputs["candidates"] = [deepcopy(inputs["candidates"][0])]  # type: ignore[index]

    def within_tolerance(origin: list[float], target: list[float]) -> float:
        distance = math.dist(origin, target)
        return distance - 0.02

    inputs["runtime_provider"].nearest_hit = within_tolerance  # type: ignore[attr-defined]
    assert _evaluate(inputs)[0]["status"] == "pass"

    def outside_tolerance(origin: list[float], target: list[float]) -> float:
        distance = math.dist(origin, target)
        return distance - 0.04

    inputs["runtime_provider"].nearest_hit = outside_tolerance  # type: ignore[attr-defined]
    record = _evaluate(inputs)[0]
    assert record["status"] == "fail"
    assert record["room_gate"] is None


def test_one_failed_actor_frame_is_blocking() -> None:
    inputs = _inputs()
    inputs["candidates"] = [deepcopy(inputs["candidates"][0])]  # type: ignore[index]
    floors = deepcopy(inputs["actor_floor_paths_m"])
    floors["source1_actor"][1][0] = -9.0
    inputs["actor_floor_paths_m"] = floors
    inputs["runtime_provider"].pathfinder.bad_actor_x = -9.0  # type: ignore[attr-defined]

    record = _evaluate(inputs)[0]
    failure = record["evidence"]["hard_gates"]["same_navmesh_island"]
    assert failure["status"] == "fail"
    assert failure["first_failure"]["frame_index"] == 1


def test_duplicate_candidates_and_incomplete_anchor_sets_fail_closed() -> None:
    duplicate = _inputs()
    duplicate["candidates"] = [
        deepcopy(duplicate["candidates"][0]),  # type: ignore[index]
        deepcopy(duplicate["candidates"][0]),  # type: ignore[index]
    ]
    with pytest.raises(CameraCandidateGateError, match="candidate IDs must be unique"):
        _evaluate(duplicate)

    incomplete = _inputs()
    incomplete["actor_visibility_anchors_m"] = deepcopy(
        incomplete["actor_visibility_anchors_m"]
    )
    incomplete["actor_visibility_anchors_m"]["source1_actor"].pop("torso")  # type: ignore[index]
    with pytest.raises(
        CameraCandidateGateError, match="at least two visibility anchors"
    ):
        _evaluate(incomplete)


@pytest.mark.parametrize(
    "field",
    ["pathfinder_loaded", "physics_enabled", "raycast_enabled"],
)
def test_runtime_provider_must_explicitly_enable_navigation_and_raycast(
    field: str,
) -> None:
    inputs = _inputs()
    provider = inputs["runtime_provider"]
    provider.runtime_context = deepcopy(provider.runtime_context)  # type: ignore[attr-defined]
    provider.runtime_context[field] = False  # type: ignore[attr-defined]
    with pytest.raises(CameraCandidateGateError, match=field):
        _evaluate(inputs)


@pytest.mark.parametrize(
    "field",
    ["floor_height_m", "maximum_snap_error_m", "minimum_clearance_m"],
)
def test_nonfinite_thresholds_are_rejected(field: str) -> None:
    inputs = _inputs()
    inputs[field] = math.nan
    with pytest.raises(CameraCandidateGateError, match="finite number"):
        _evaluate(inputs)


def test_runtime_pathfinder_and_each_full_ray_are_provider_bound() -> None:
    unloaded = _inputs()
    unloaded["runtime_provider"].pathfinder.is_loaded = False  # type: ignore[attr-defined]
    with pytest.raises(CameraCandidateGateError, match="pathfinder.is_loaded"):
        _evaluate(unloaded)

    drift = _inputs()
    provider = drift["runtime_provider"]
    original = provider.line_of_sight_nearest_hit  # type: ignore[attr-defined]

    def wrong_scene(origin: list[float], target: list[float]) -> dict[str, object]:
        evidence = original(origin, target)
        evidence["scene_id"] = "another-scene"
        return evidence

    provider.line_of_sight_nearest_hit = wrong_scene  # type: ignore[attr-defined]
    with pytest.raises(CameraCandidateGateError, match="scene_id differs"):
        _evaluate(drift)


def test_endpoint_policy_must_be_full_ray_without_provider_tolerance() -> None:
    inputs = _inputs()
    provider = inputs["runtime_provider"]
    original = provider.line_of_sight_nearest_hit  # type: ignore[attr-defined]

    def shortened_ray(origin: list[float], target: list[float]) -> dict[str, object]:
        evidence = original(origin, target)
        evidence["endpoint_policy"] = "shortened_by_provider"
        return evidence

    provider.line_of_sight_nearest_hit = shortened_ray  # type: ignore[attr-defined]
    with pytest.raises(CameraCandidateGateError, match="endpoint_policy differs"):
        _evaluate(inputs)


class FakeRay:
    def __init__(self, origin: np.ndarray, direction: np.ndarray) -> None:
        self.origin = np.asarray(origin, dtype=np.float64)
        self.direction = np.asarray(direction, dtype=np.float64)


class FakeHabitatSimulator:
    def __init__(self) -> None:
        self.pathfinder = SimpleNamespace(is_loaded=True)
        self.config = SimpleNamespace(
            sim_cfg=SimpleNamespace(enable_physics=True, scene_id="mp3d-room")
        )
        self.curr_scene_name = "mp3d-room"
        self.active_dataset = "mp3d-dataset"
        self.scene_aabb = ([-4.0, -1.0, -5.0], [4.0, 4.0, 3.0])
        self.hits: list[object] = []
        self.last_cast: tuple[object, float, float] | None = None

    def get_physics_simulation_library(self) -> str:
        return "bullet"

    def get_stage_initialization_template(self) -> object:
        return SimpleNamespace(render_asset_fullpath="/scenes/mp3d-room.glb")

    def cast_ray(
        self, ray: object, *, max_distance: float, buffer_distance: float
    ) -> object:
        self.last_cast = (ray, max_distance, buffer_distance)
        return SimpleNamespace(has_hits=lambda: bool(self.hits), hits=self.hits)


def _real_provider(simulator: FakeHabitatSimulator) -> HabitatRuntimeCameraProvider:
    habitat = SimpleNamespace(geo=SimpleNamespace(Ray=FakeRay))
    magnum = SimpleNamespace(Vector3=lambda value: np.asarray(value, dtype=np.float64))
    return HabitatRuntimeCameraProvider(
        simulator,
        habitat,
        magnum,
        "mp3d-room",
        provider_id="mp3d-runtime-01",
    )


def test_habitat_provider_uses_loaded_scene_bounds_and_complete_zero_buffer_ray() -> (
    None
):
    simulator = FakeHabitatSimulator()
    provider = _real_provider(simulator)

    assert provider.pathfinder is simulator.pathfinder
    assert provider.room_bounds_m == {
        "minimum_m": [-4.0, -1.0, -5.0],
        "maximum_m": [4.0, 4.0, 3.0],
    }
    assert provider.runtime_context["room_bounds_source"] == "loaded_scene_aabb"
    assert provider.runtime_context["active_dataset"] == "mp3d-dataset"
    assert provider.runtime_context["stage_surface"] == "/scenes/mp3d-room.glb"

    evidence = provider.line_of_sight_nearest_hit([0.0, 1.0, 0.0], [3.0, 1.0, 4.0])
    ray, max_distance, buffer_distance = simulator.last_cast  # type: ignore[misc]
    assert np.allclose(ray.origin, [0.0, 1.0, 0.0])
    assert np.allclose(ray.direction, [0.6, 0.0, 0.8])
    assert max_distance == 5.0
    assert buffer_distance == 0.0
    assert evidence["nearest_hit_distance_m"] is None
    assert evidence["endpoint_policy"] == ("full_ray_buffer_zero_no_endpoint_tolerance")


def test_habitat_provider_returns_nearest_actual_hit() -> None:
    simulator = FakeHabitatSimulator()
    simulator.hits = [
        SimpleNamespace(
            ray_distance=3.0,
            object_id=9,
            point=[0.0, 1.0, -3.0],
            normal=[0.0, 0.0, 1.0],
        ),
        SimpleNamespace(
            ray_distance=1.5,
            object_id=4,
            point=[0.0, 1.0, -1.5],
            normal=[0.0, 0.0, 1.0],
        ),
    ]
    evidence = _real_provider(simulator).line_of_sight_nearest_hit(
        [0.0, 1.0, 0.0], [0.0, 1.0, -4.0]
    )

    assert evidence["nearest_hit_distance_m"] == 1.5
    assert evidence["nearest_hit"] == {
        "object_id": 4,
        "point_m": [0.0, 1.0, -1.5],
        "normal": [0.0, 0.0, 1.0],
    }


@pytest.mark.parametrize("mutation", ["physics", "scene", "pathfinder"])
def test_habitat_provider_fails_closed_on_runtime_mismatch(mutation: str) -> None:
    simulator = FakeHabitatSimulator()
    if mutation == "physics":
        simulator.config.sim_cfg.enable_physics = False
    elif mutation == "scene":
        simulator.curr_scene_name = "different-room"
    else:
        simulator.pathfinder.is_loaded = False

    with pytest.raises(CameraCandidateGateError):
        _real_provider(simulator)


def test_habitat_provider_supports_distinct_configured_and_loaded_scene_ids() -> None:
    simulator = FakeHabitatSimulator()
    simulator.config.sim_cfg.scene_id = "/datasets/mp3d/room.glb"
    simulator.curr_scene_name = "mp3d-room-instance"
    habitat = SimpleNamespace(geo=SimpleNamespace(Ray=FakeRay))
    magnum = SimpleNamespace(Vector3=lambda value: np.asarray(value, dtype=np.float64))

    provider = HabitatRuntimeCameraProvider(
        simulator,
        habitat,
        magnum,
        {
            "configured_scene_id": "/datasets/mp3d/room.glb",
            "loaded_scene_id": "mp3d-room-instance",
            "active_dataset": "mp3d-dataset",
            "stage_surface": "/scenes/mp3d-room.glb",
        },
    )

    assert provider.runtime_context["configured_scene_id"] == (
        "/datasets/mp3d/room.glb"
    )
    assert provider.runtime_context["scene_id"] == "mp3d-room-instance"

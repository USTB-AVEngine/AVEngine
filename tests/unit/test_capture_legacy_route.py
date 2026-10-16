from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pytest

from avengine.contracts.json_io import canonical_json_sha256
from avengine.capture.legacy_route import (
    FRAME_COUNT,
    SSOT_TO_HABITAT_EQUATION,
    evaluate_center_point_gate,
    ssot_point_to_habitat,
    ssot_yaw_to_habitat_yaw,
    validate_route_manifest,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
EXAMPLE_PATH = REPOSITORY_ROOT / "examples/capture/legacy_apartment/route_manifest.json"
LEGACY_SPEC_SHA256 = "7934a2eb57b838b176b5151baa2b88e43c8c69cfe377ca3fa8a8edc12e85d909"
LEGACY_270_POINT_SHA256 = (
    "8138e5494c63eb8352b73752f13b443d456fc623432c8c7f1f4591d35e249f67"
)


@pytest.fixture(scope="module")
def manifest() -> dict:
    return json.loads(EXAMPLE_PATH.read_text(encoding="utf-8"))


def test_frozen_ssot_to_habitat_transform_camera_yaw_and_fov(manifest: dict) -> None:
    assert manifest["coordinate_transform"]["equation"] == SSOT_TO_HABITAT_EQUATION
    assert ssot_point_to_habitat([0.5, 0.15, 1.2]) == [-0.7, 1.471, 0.65]
    assert ssot_yaw_to_habitat_yaw(145.0) == 55.0
    assert manifest["camera"] == {
        "camera_id": "view0",
        "habitat_position_m": [-0.7, 1.471, 0.65],
        "habitat_yaw_deg": 55.0,
        "horizontal_fov_deg": 105.0,
        "ssot_position_m": [0.5, 0.15, 1.2],
        "ssot_yaw_deg": 145.0,
    }


def test_example_closes_legacy_input_and_both_270_point_routes(manifest: dict) -> None:
    assert validate_route_manifest(manifest) == []
    assert manifest["status"] == "pass"
    assert manifest["timebase"] == {
        "duration_seconds": 18.0,
        "frame_count": 270,
        "frame_rate_hz": 15,
    }
    authority = manifest["authoritative_legacy_input"]
    assert authority["sha256"] == LEGACY_SPEC_SHA256
    assert authority["trajectory_point_count"] == FRAME_COUNT
    assert authority["ssot_trajectory_sha256"] == LEGACY_270_POINT_SHA256

    for route in manifest["routes"].values():
        assert route["point_count"] == FRAME_COUNT
        assert len(route["ssot_trajectory_m"]) == FRAME_COUNT
        assert len(route["habitat_trajectory_m"]) == FRAME_COUNT
        assert route["ssot_trajectory_sha256"] == canonical_json_sha256(
            route["ssot_trajectory_m"]
        )
        assert route["habitat_trajectory_sha256"] == canonical_json_sha256(
            route["habitat_trajectory_m"]
        )


def test_every_habitat_point_is_the_declared_ssot_transform(manifest: dict) -> None:
    for route in manifest["routes"].values():
        expected = [
            ssot_point_to_habitat(point) for point in route["ssot_trajectory_m"]
        ]
        assert route["habitat_trajectory_m"] == expected
    assert manifest["routes"]["human_path"]["habitat_trajectory_m"][0] == [
        2.380148,
        0.271,
        1.830123,
    ]
    assert manifest["routes"]["human_path"]["habitat_trajectory_m"][-1] == [
        -2.83,
        0.271,
        pytest.approx(-1.75),
    ]


def test_all_legacy_aabbs_are_migrated_and_navmesh_is_not_the_gate(
    manifest: dict,
) -> None:
    obstacles = manifest["obstacles"]
    assert len(obstacles) == 57
    assert sum(item["included_in_point_gate"] for item in obstacles) == 41
    assert (
        sum(
            item["source_kind"] == "legacy_apartment_furniture_map"
            for item in obstacles
        )
        == 33
    )
    assert (
        sum(item["source_kind"] == "legacy_apartment_shell_map" for item in obstacles)
        == 21
    )
    assert (
        sum(
            item["source_kind"] == "legacy_builtin_visual_obstacle"
            for item in obstacles
        )
        == 3
    )
    navmesh = manifest["obstacle_policy"]["navmesh_diagnostic"]
    assert navmesh["agent_radius_m"] == 0.2
    assert navmesh["formal_gate"] is False
    for obstacle in obstacles:
        habitat = obstacle["bbox_habitat_m"]
        assert obstacle["horizontal_aabb_habitat_xz_m"] == {
            "minimum": [habitat["minimum"][0], habitat["minimum"][2]],
            "maximum": [habitat["maximum"][0], habitat["maximum"][2]],
        }


def test_per_frame_center_point_gates_and_clearances_are_authoritative(
    manifest: dict,
) -> None:
    human = manifest["gates"]["human_center_point_aabb"]
    dog = manifest["gates"]["dog_center_point_aabb"]
    for gate in (human, dog):
        assert gate["collision_primitive"] == "zero_radius_center_point_habitat_xz"
        assert gate["agent_radius_m"] == 0.0
        assert gate["navmesh_is_gate"] is False
        assert gate["status"] == "pass"
        assert gate["collision_count"] == 0
        assert len(gate["frames"]) == FRAME_COUNT
        assert all(frame["status"] == "pass" for frame in gate["frames"])
        assert gate["minimum_clearance_m"] == min(
            frame["minimum_clearance_m"] for frame in gate["frames"]
        )
    assert human["minimum_clearance_m"] == pytest.approx(0.2)
    assert dog["minimum_clearance_m"] == pytest.approx(0.09956543726992528)


def test_dog_reverse_offset_is_safe_and_centers_never_collide(manifest: dict) -> None:
    human = np.asarray(
        manifest["routes"]["human_path"]["ssot_trajectory_m"], dtype=np.float64
    )
    dog = np.asarray(
        manifest["routes"]["dog_path"]["ssot_trajectory_m"], dtype=np.float64
    )
    assert np.array_equal(dog, human[::-1] + np.asarray([-0.35, 0.0, 0.0]))
    separation = manifest["gates"]["inter_source_center_separation"]
    assert separation["status"] == "pass"
    assert separation["collision_count"] == 0
    assert separation["minimum_required_m"] == 0.3
    assert separation["minimum_observed_m"] == pytest.approx(0.35907867488337414)
    assert len(separation["per_frame_separation_m"]) == FRAME_COUNT


def test_direct_point_gate_catches_a_center_inside_an_aabb(manifest: dict) -> None:
    route = deepcopy(manifest["routes"]["human_path"]["habitat_trajectory_m"])
    obstacle = next(
        item for item in manifest["obstacles"] if item["included_in_point_gate"]
    )
    horizontal = obstacle["horizontal_aabb_habitat_xz_m"]
    route[0] = [
        0.5 * (horizontal["minimum"][0] + horizontal["maximum"][0]),
        0.271,
        0.5 * (horizontal["minimum"][1] + horizontal["maximum"][1]),
    ]
    report = evaluate_center_point_gate(route, manifest["obstacles"])
    assert report["status"] == "fail"
    assert report["collision_count"] >= 1
    assert report["frames"][0]["status"] == "fail"
    assert report["minimum_clearance_m"] == 0.0
    assert report["agent_radius_m"] == 0.0


def test_semantic_validator_rejects_route_and_aabb_tampering(manifest: dict) -> None:
    changed_route = deepcopy(manifest)
    changed_route["routes"]["dog_path"]["habitat_trajectory_m"][0][0] += 0.01
    errors = validate_route_manifest(changed_route)
    assert "manifest_content_sha256 differs" in errors
    assert "dog_path Habitat trajectory hash differs" in errors
    assert "dog_path Habitat points differ from SSOT transform" in errors

    changed_aabb = deepcopy(manifest)
    changed_aabb["obstacles"][0]["horizontal_aabb_habitat_xz_m"]["minimum"][0] += 0.01
    errors = validate_route_manifest(changed_aabb)
    assert "obstacle 0 horizontal point-gate AABB differs" in errors

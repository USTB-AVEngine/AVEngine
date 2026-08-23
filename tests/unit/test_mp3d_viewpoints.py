from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from avengine.contracts.json_io import load_json
from avengine.m5.mp3d_viewpoints import (
    MP3DViewpointPrefilterError,
    select_mp3d_viewpoints,
)
from tools.m5 import build_mp3d_viewpoint_bank as builder


REPOSITORY = Path(__file__).resolve().parents[2]
ROOM_MANIFEST = REPOSITORY / "examples/m1/rooms/habitat_mp3d_example/room_manifest.json"
BASE_REQUEST = REPOSITORY / "examples/m1/requests/habitat_mp3d_example.json"
ROOM_ID = "habitat_mp3d_example_17DRP5sb8fy"


def _routes() -> dict[str, list[list[float]]]:
    x = np.linspace(-1.5, 1.5, 15)
    return {
        "actor0": np.stack(
            (x, np.zeros_like(x), np.full_like(x, -4.0)), axis=1
        ).tolist(),
        "actor1": np.stack(
            (x + 0.25, np.zeros_like(x), np.full_like(x, -5.0)), axis=1
        ).tolist(),
    }


def _floor_samples() -> list[list[float]]:
    center = np.asarray([0.0, 0.0, -4.5], dtype=np.float64)
    result = []
    for radius in (4.5, 5.5):
        for degrees in range(0, 360, 30):
            radians = np.radians(degrees)
            result.append(
                (
                    center
                    + np.asarray(
                        [radius * np.sin(radians), 0.0, radius * np.cos(radians)]
                    )
                ).tolist()
            )
    return result


def _select(*, count: int = 6, floor_samples=None, routes=None):
    return select_mp3d_viewpoints(
        room_id=ROOM_ID,
        room_navigation={"agent_height_m": 1.5},
        base_m1_request=load_json(BASE_REQUEST),
        camera_floor_samples_m=(
            _floor_samples() if floor_samples is None else floor_samples
        ),
        actor_routes_m=_routes() if routes is None else routes,
        selection_count=count,
    )


def test_selection_is_deterministic_and_input_order_independent() -> None:
    floor_samples = _floor_samples()
    routes = _routes()
    first = _select(floor_samples=floor_samples, routes=routes)
    second = _select(
        floor_samples=list(reversed(floor_samples)),
        routes={key: routes[key] for key in reversed(routes)},
    )

    assert first.to_record() == second.to_record()
    assert [item.m1_request for item in first.viewpoints] == [
        item.m1_request for item in second.viewpoints
    ]


@pytest.mark.parametrize("count", [4, 6, 8])
def test_selector_returns_requested_four_to_eight_distinct_candidates(
    count: int,
) -> None:
    result = _select(count=count)

    assert len(result.viewpoints) == count
    assert all(not item["near_duplicate"] for item in result.pairwise_deduplication)
    assert len(result.pairwise_deduplication) == count * (count - 1) // 2
    assert all(
        viewpoint.m1_request["primary_camera_rig"]["view_id"] == "view0"
        for viewpoint in result.viewpoints
    )


def test_near_duplicate_floor_samples_do_not_repeat_a_selected_view() -> None:
    samples = _floor_samples()
    samples.extend(
        [
            [samples[0][0] + 0.01, samples[0][1], samples[0][2]],
            [samples[0][0] + 0.02, samples[0][1], samples[0][2] + 0.01],
        ]
    )
    result = _select(count=8, floor_samples=samples)

    assert all(not item["near_duplicate"] for item in result.pairwise_deduplication)


def test_tiny_authority_fails_when_four_distinct_views_are_impossible() -> None:
    repeated_route = {
        "actor0": [[0.0, 0.0, -4.0], [0.0, 0.0, -4.0]],
        "actor1": [[0.0, 0.0, -4.0], [0.0, 0.0, -4.0]],
    }
    with pytest.raises(
        MP3DViewpointPrefilterError, match="distinct analytic candidates"
    ):
        _select(
            count=4,
            floor_samples=[[0.0, 0.0, 0.0]],
            routes=repeated_route,
        )


@pytest.mark.parametrize("count", [3, 9])
def test_selection_count_outside_four_to_eight_is_rejected(count: int) -> None:
    with pytest.raises(MP3DViewpointPrefilterError, match="within 4--8"):
        _select(count=count)


def test_each_candidate_retains_an_explicit_per_actor_route_subset() -> None:
    result = _select(count=6)

    for viewpoint in result.viewpoints:
        assert set(viewpoint.route_subset) == {"actor0", "actor1"}
        for subset in viewpoint.route_subset.values():
            assert subset["visible_frame_count"] > 0
            assert subset["visible_frame_ranges_inclusive"]
            assert subset["visible_segment_indices"]


def _write_builder_inputs(tmp_path: Path) -> tuple[Path, Path]:
    route = tmp_path / "route_explanation.json"
    route.write_text(
        json.dumps(
            {
                "research_only": True,
                "episode_counted": False,
                "provenance": {"new_request_room_id": ROOM_ID},
                "native_pathfinder": {
                    "all_paths_share_single_island": True,
                    "island_id": 7,
                },
                "current_visual_realization": {"actor_skin_root_paths_m": _routes()},
            }
        ),
        encoding="utf-8",
    )
    floor = tmp_path / "camera_floor_samples.json"
    floor.write_text(
        json.dumps(
            {
                "room_id": ROOM_ID,
                "authority": "fixture_native_pathfinder_floor_samples",
                "camera_floor_samples_m": _floor_samples(),
            }
        ),
        encoding="utf-8",
    )
    return route, floor


def test_builder_publishes_fresh_requests_with_pending_native_claims(
    tmp_path: Path,
) -> None:
    route, floor = _write_builder_inputs(tmp_path)
    output = tmp_path / "fresh-mp3d-viewpoints"
    args = builder.parse_args(
        [
            "--room-manifest",
            str(ROOM_MANIFEST),
            "--base-m1-request",
            str(BASE_REQUEST),
            "--route-explanation",
            str(route),
            "--camera-floor-samples",
            str(floor),
            "--selection-count",
            "6",
            "--output",
            str(output),
        ]
    )

    assert builder.run(args) == output
    bank = json.loads((output / "viewpoint_bank.json").read_text(encoding="utf-8"))
    assert bank["artifact_status"] == "cpu_offline_candidate_prefilter_complete"
    assert bank["certification_status"] == "pending_native_habitat"
    assert bank["native_candidate_navigation_validation_status"] == "pending"
    assert bank["native_habitat_pixel_status"] == "pending"
    assert bank["first_person_certification_status"] == "pending"
    assert bank["four_point_mini_pilot_status"] == "pending"
    assert bank["research_only"] is True
    assert bank["episode_counted"] is False
    assert len(bank["viewpoints"]) == 6
    assert len(bank["m1_capture_request_paths"]) == 6

    for viewpoint in bank["viewpoints"]:
        request = json.loads(
            (
                output / bank["m1_capture_request_paths"][viewpoint["viewpoint_id"]]
            ).read_text(encoding="utf-8")
        )
        assert request["primary_camera_rig"]["view_id"] == "view0"
        assert request["listener"] == {
            "listener_id": "listener0",
            "attached_to": "camera_rig_0",
            "rig_from_listener": {
                "translation_m": [0.0, 0.0, 0.0],
                "rotation_xyzw": [0.0, 0.0, 0.0, 1.0],
            },
        }

    with pytest.raises(builder.MP3DViewpointBuildError, match="refusing to replace"):
        builder.run(args)


def test_builder_rejects_route_authority_from_another_room(tmp_path: Path) -> None:
    route, floor = _write_builder_inputs(tmp_path)
    value = json.loads(route.read_text(encoding="utf-8"))
    value["provenance"]["new_request_room_id"] = "another_room"
    route.write_text(json.dumps(value), encoding="utf-8")
    args = builder.parse_args(
        [
            "--room-manifest",
            str(ROOM_MANIFEST),
            "--base-m1-request",
            str(BASE_REQUEST),
            "--route-explanation",
            str(route),
            "--camera-floor-samples",
            str(floor),
            "--output",
            str(tmp_path / "fresh"),
        ]
    )

    with pytest.raises(builder.MP3DViewpointBuildError, match="room differs"):
        builder.run(args)

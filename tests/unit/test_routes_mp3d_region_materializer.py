from __future__ import annotations

import json
from pathlib import Path

import pytest

from avengine.registry.sources import load_source_endpoint_registry
from avengine.rooms.contracts import load_and_validate_inputs as load_m1_inputs
from avengine.routes.mp3d_region_materializer import (
    MP3DRegionMaterializationError,
    materialize_region_case,
)
from avengine.timeline.current_mp3d_dynamic_audio import (
    load_captured_render_clock,
    load_captured_source_paths,
)


REPOSITORY = Path(__file__).resolve().parents[2]
ROOM_MANIFEST = REPOSITORY / "examples/rooms/habitat_mp3d_example/room_manifest.json"
M1_REQUEST = REPOSITORY / "examples/rooms/requests/habitat_mp3d_example.json"
ENDPOINT_REGISTRY = REPOSITORY / "examples/registry/registries/source_endpoints_v1.json"
HUMAN_ASSET_0 = "rocketbox_dog_beagle_01_m2_v7_world_contact_candidate"
HUMAN_ASSET_1 = "rocketbox_dog_beagle_01_m2_v7_world_contact_candidate"
ASSET_REVISION = "m2_v7_world_contact_r5"


def _minimal_house() -> str:
    return "\n".join(
        [
            "ASCII 1.1",
            "H - - 0 0 4 1 0 0 0 1 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0",
            "L 0 0 - 0 0 0 -1 -1 -1 1 1 1 0 0 0 0 0",
            "R 0 0 0 0 l 0 0 0 -1 -0.1 -1 1 0.1 1 0 0 0 0 0",
            "S 0 0 0 F 0 0 0 0 0 1 -1 -1 -1 1 1 1 0 0 0 0 0",
            "V 0 0 F -1 -1 0 0 0 1 0 0 0 0",
            "V 1 0 F 1 -1 0 0 0 1 0 0 0 0",
            "V 2 0 F 1 1 0 0 0 1 0 0 0 0",
            "V 3 0 F -1 1 0 0 0 1 0 0 0 0",
        ]
    ) + "\n"


def _selection(tmp_path: Path) -> Path:
    value = {
        "schema": "avengine_n_actor_selection_v1",
        "research_only": True,
        "episode_counted": False,
        "actors": [
            {
                "source_slot_id": "source1",
                "asset_id": HUMAN_ASSET_0,
                "revision": ASSET_REVISION,
                "legacy_timeline_actor_id": "beagle_0",
            },
            {
                "source_slot_id": "source2",
                "asset_id": HUMAN_ASSET_1,
                "revision": ASSET_REVISION,
                "legacy_timeline_actor_id": "beagle_1",
            },
        ],
    }
    path = tmp_path / "actor_selection.json"
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def _plan(tmp_path: Path, *, frame_rate_hz: int = 15) -> Path:
    house = tmp_path / "17DRP5sb8fy.house"
    house.write_text(_minimal_house(), encoding="utf-8")
    family_id = "17DRP5sb8fy_region_000_route_family_01"
    case = {
        "route_family_id": family_id,
        "motion_case": "both_moving",
        "frame_count": 3,
        "frame_rate_hz": frame_rate_hz,
        "source1_positions_m": [
            [-0.6, 0.0, -0.6],
            [0.0, 0.0, 0.0],
            [0.6, 0.0, 0.6],
        ],
        "source2_positions_m": [
            [0.6, 0.0, -0.6],
            [0.0, 0.0, 0.4],
            [-0.6, 0.0, 0.6],
        ],
    }
    static_case = json.loads(json.dumps(case))
    static_case["motion_case"] = "static_static"
    static_case["source1_positions_m"] = [case["source1_positions_m"][0]] * 3
    static_case["source2_positions_m"] = [case["source2_positions_m"][0]] * 3
    value = {
        "artifact_kind": "mp3d_region_source_route_plan",
        "research_only": True,
        "episode_counted": False,
        "house_id": "17DRP5sb8fy",
        "inputs": {"house": str(house)},
        "parameters": {
            "frame_count": 3,
            "frame_rate_hz": frame_rate_hz,
            "maximum_y_delta_m": 0.3,
        },
        "regions": [
            {
                "region_index": 0,
                "region_instance_id": "17DRP5sb8fy:region:000",
                "category_code": "l",
                "category_name": "living room",
                "route_families": [
                    {
                        "route_family_id": family_id,
                        "region_index": 0,
                        "camera_binding": {
                            "placement_id": "fixture_camera",
                            "region_index": 0,
                            "region_instance_id": "17DRP5sb8fy:region:000",
                            "floor_position_m": [-0.5, 0.0, 0.0],
                            "position_m": [-0.5, 1.2, 0.0],
                            "yaw_deg": 0.0,
                        },
                        "cases": {
                            "both_moving": case,
                            "static_static": static_case,
                        },
                    }
                ],
            }
        ],
    }
    path = tmp_path / "region_plan.json"
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def _parser_probe(
    materialized: Path, receipt: dict, tmp_path: Path
) -> Path:
    """Adapt planned records only in a test probe; native capture stays absent."""

    planned = json.loads(
        (materialized / "planned_frame_records.json").read_text(encoding="utf-8")
    )
    probe = tmp_path / "parser-probe"
    probe.mkdir()
    observed_shape = {
        "render": planned["render"],
        "frames": [
            {
                "frame_index": frame["frame_index"],
                "pts_ticks": frame["pts_ticks"],
                "source_positions_m": frame["planned_source_positions_m"],
            }
            for frame in planned["frames"]
        ],
    }
    (probe / "frame_records.json").write_text(
        json.dumps(observed_shape), encoding="utf-8"
    )
    (probe / "research_receipt.json").write_text(
        json.dumps({"capture": receipt["planned_clock"]}), encoding="utf-8"
    )
    return probe


def test_materialize_region_case_writes_inputs_readable_by_current_parsers(
    tmp_path: Path,
) -> None:
    output = tmp_path / "materialized"
    receipt = materialize_region_case(
        _plan(tmp_path),
        room_manifest_path=ROOM_MANIFEST,
        m1_request_path=M1_REQUEST,
        actor_selection_path=_selection(tmp_path),
        source_endpoint_registry_path=ENDPOINT_REGISTRY,
        output_directory=output,
        region_index=0,
        motion_case="both_moving",
        frame_count=5,
        frame_rate_hz=15,
    )

    assert receipt["status"] == "research_only"
    assert receipt["planned_clock"]["frame_count"] == 5
    assert receipt["planned_clock"]["sample_count"] == 5333
    assert receipt["audio"]["status"] == "requires_explicit_audio_program"
    assert receipt["native_capture"]["status"] == "not_run"
    assert not (output / "frame_records.json").exists()
    m1 = load_m1_inputs(ROOM_MANIFEST, output / "m1_capture_request.json")
    assert m1.request["room_id"] == "habitat_mp3d_example_17DRP5sb8fy"
    assert len(m1.request["sources"]) == 2
    endpoints = load_source_endpoint_registry(output / "source_endpoints.json")
    endpoint_ids = tuple(receipt["audio"]["source_endpoint_ids"])
    assert endpoint_ids == ("beagle_0_muzzle", "beagle_1_muzzle")
    assert {item["source_endpoint_id"] for item in endpoints["source_endpoints"]} >= set(endpoint_ids)
    planned = json.loads(
        (output / "planned_frame_records.json").read_text(encoding="utf-8")
    )
    assert planned["artifact_role"] == "planned_frame_records_not_observed_capture"
    assert "planned_source_positions_m" in planned["frames"][0]
    assert "source_positions_m" not in planned["frames"][0]
    probe = _parser_probe(output, receipt, tmp_path)
    clock = load_captured_render_clock(probe)
    assert clock["frame_count"] == 5
    assert clock["frame_rate_hz"] == 15
    assert clock["sample_count"] == 5333
    trajectories = load_captured_source_paths(probe, endpoint_ids)
    assert all(len(points) == 5 for points in trajectories.values())
    assert json.loads((output / "planned_timeline.json").read_text())["render"]["frame_count"] == 5


def test_materialize_region_case_holds_static_routes_at_any_output_clock(
    tmp_path: Path,
) -> None:
    output = tmp_path / "static-materialized"
    receipt = materialize_region_case(
        _plan(tmp_path),
        room_manifest_path=ROOM_MANIFEST,
        m1_request_path=M1_REQUEST,
        actor_selection_path=_selection(tmp_path),
        source_endpoint_registry_path=ENDPOINT_REGISTRY,
        output_directory=output,
        region_index=0,
        motion_case="static_static",
        frame_count=150,
        frame_rate_hz=15,
    )
    ids = tuple(receipt["audio"]["source_endpoint_ids"])
    probe = _parser_probe(output, receipt, tmp_path)
    trajectories = load_captured_source_paths(probe, ids)
    for points in trajectories.values():
        assert points == [points[0]] * 150


def test_materializer_requires_a_current_compatible_clock(tmp_path: Path) -> None:
    with pytest.raises(
        MP3DRegionMaterializationError,
        match="incompatible.*explicit",
    ):
        materialize_region_case(
            _plan(tmp_path, frame_rate_hz=11),
            room_manifest_path=ROOM_MANIFEST,
            m1_request_path=M1_REQUEST,
            actor_selection_path=_selection(tmp_path),
            source_endpoint_registry_path=ENDPOINT_REGISTRY,
            output_directory=tmp_path / "incompatible",
            region_index=0,
            motion_case="both_moving",
        )


def test_materializer_checks_optional_audio_program_endpoint_and_clock(
    tmp_path: Path,
) -> None:
    audio_program = tmp_path / "audio_program.json"
    audio_program.write_text(
        json.dumps(
            {
                "schema": "avengine_m6_audio_program_v1",
                "candidate_source_endpoint_ids": [
                    "beagle_0_muzzle",
                    "beagle_1_muzzle",
                ],
                "timeline": {
                    "time_base_hz": 48000,
                    "ticks_per_frame": 3200,
                    "video_fps": 15,
                    "frame_count": 5,
                    "sample_rate_hz": 16000,
                    "ticks_per_sample": 3,
                    "sample_count": 5333,
                },
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "with-audio-input"
    receipt = materialize_region_case(
        _plan(tmp_path),
        room_manifest_path=ROOM_MANIFEST,
        m1_request_path=M1_REQUEST,
        actor_selection_path=_selection(tmp_path),
        source_endpoint_registry_path=ENDPOINT_REGISTRY,
        audio_program_path=audio_program,
        output_directory=output,
        region_index=0,
        motion_case="both_moving",
        frame_count=5,
        frame_rate_hz=15,
    )
    assert receipt["audio"]["status"] == "planned_program_clock_and_endpoint_bound"
    assert (output / "audio_program.json").is_file()

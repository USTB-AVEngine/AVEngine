from __future__ import annotations

import json
from pathlib import Path

import pytest

from avengine.assets.mp3d_region_actor_tracks import (
    MP3DRegionActorTrackError,
    materialize_region_actor_tracks,
)
from avengine.rooms.contracts import load_and_validate_inputs as load_m1_inputs


REPOSITORY = Path(__file__).resolve().parents[2]
ROOM_MANIFEST = REPOSITORY / "examples/rooms/habitat_mp3d_example/room_manifest.json"
BEAGLE_ASSET = Path(
    "/data/avengine_external/datasets/m2/"
    "rocketbox_beagle_m2_canary_v7_world_contact_r5/asset_manifest.json"
)
BEAGLE_REQUEST = Path(
    "/data/avengine_external/review/"
    "current_mp3d_two_beagle_route_lateral_seed22_grounded/primary_m2_request.json"
)
HUMAN_GLB = Path(
    "/data/avengine_external/datasets/m5_1/"
    "rocketbox_male_adult_01_original_ue_v3_20260820T235905Z/runtime.glb"
)
ASSET_ID = "rocketbox_dog_beagle_01_m2_v7_world_contact_candidate"
ASSET_REVISION = "m2_v7_world_contact_r5"


def _write_m1(tmp_path: Path, endpoint_ids: list[str]) -> Path:
    request = json.loads(
        (REPOSITORY / "examples/rooms/requests/habitat_mp3d_example.json")
        .read_text(encoding="utf-8")
    )
    base_positions = [
        [-7.0, 0.072447, -3.0],
        [-8.0, 0.072447, -4.0],
        [-9.0, 0.072447, -5.0],
    ]
    request["sources"] = [
        {
            "source_id": endpoint_id,
            "world_from_source": {
                "translation_m": base_positions[index],
                "rotation_xyzw": [0.0, 0.0, 0.0, 1.0],
            },
        }
        for index, endpoint_id in enumerate(endpoint_ids)
    ]
    path = tmp_path / "m1_capture_request.json"
    path.write_text(json.dumps(request), encoding="utf-8")
    return path


def _write_actor_config(tmp_path: Path, actor_count: int) -> Path:
    actors = []
    for index in range(actor_count):
        actors.append(
            {
                "actor_id": f"beagle_{index}",
                "source_slot_id": f"source{index + 1}",
                "source_endpoint_id": f"source{index + 1}",
                "semantic_id": 210 + index,
                "asset_id": ASSET_ID,
                "asset_revision": ASSET_REVISION,
                "asset_manifest_path": str(BEAGLE_ASSET),
                "base_m2_request_path": str(BEAGLE_REQUEST),
                "emitter_anchor_id": "muzzle",
                "route_to_actor_root_offset_m": [0.0, 0.0, 0.0],
            }
        )
    path = tmp_path / f"actor_config_{actor_count}.json"
    path.write_text(json.dumps({"actors": actors}), encoding="utf-8")
    return path


def _write_planned_inputs(tmp_path: Path, actor_count: int, *, frame_rate_hz: int = 15) -> Path:
    endpoint_ids = [f"source{index + 1}" for index in range(actor_count)]
    positions = [
        [
            [-7.0 - index, 0.072447, -3.0 - index],
            [-6.5 - index, 0.072447, -3.0 - index],
            [-6.0 - index, 0.072447, -3.0 - index],
            [-5.5 - index, 0.072447, -3.0 - index],
            [-5.0 - index, 0.072447, -3.0 - index],
        ]
        for index in range(actor_count)
    ]
    family_id = "17DRP5sb8fy_region_000_route_family_01"
    actors = [
        {
            "source_slot_id": f"source{index + 1}",
            "actor_id": f"beagle_{index}",
            "asset_id": ASSET_ID,
            "revision": ASSET_REVISION,
            "source_endpoint_id": endpoint_ids[index],
        }
        for index in range(actor_count)
    ]
    frames = []
    for frame_index in range(5):
        frames.append(
            {
                "frame_index": frame_index,
                "pts_ticks": frame_index * 3200,
                "planned_camera_pose": {
                    "position_m": [-9.0, 1.572447, -2.0],
                    "yaw_deg": 0.0,
                },
                "actor_states": [
                    {
                        **actor,
                        "planned_route_center_m": positions[index][frame_index],
                        "position_semantics": "planner_route_center_not_emitter_readback",
                        "action_id": "walk",
                        "action_phase": frame_index / 4.0,
                    }
                    for index, actor in enumerate(actors)
                ],
            }
        )
    timeline = {
        "schema": "avengine_mp3d_region_planned_timeline_v1",
        "artifact_role": "planned_timeline_not_native_capture",
        "status": "research_only",
        "research_only": True,
        "episode_counted": False,
        "qualification_claim": False,
        "room": {
            "room_id": "habitat_mp3d_example_17DRP5sb8fy",
            "house_id": "17DRP5sb8fy",
        },
        "region": {
            "region_index": 0,
            "region_instance_id": "17DRP5sb8fy:region:000",
        },
        "route_family_id": family_id,
        "motion_case": "both_moving",
        "render": {
            "frame_count": 5,
            "frame_rate_hz": frame_rate_hz,
            "ticks_per_frame": 3200,
            "time_base_hz": 48000,
            "sample_rate_hz": 16000,
            "sample_count": 5333,
        },
        "actors": actors,
        "frames": frames,
    }
    timeline_path = tmp_path / f"planned_timeline_{actor_count}.json"
    timeline_path.write_text(json.dumps(timeline), encoding="utf-8")
    plan = {
        "artifact_kind": "mp3d_region_source_route_plan",
        "research_only": True,
        "episode_counted": False,
        "house_id": "17DRP5sb8fy",
        "regions": [
            {
                "region_index": 0,
                "region_instance_id": "17DRP5sb8fy:region:000",
                "route_families": [
                    {
                        "route_family_id": family_id,
                        "cases": {"both_moving": {
                            **{
                                f"source{index + 1}_positions_m": positions[index]
                                for index in range(actor_count)
                            },
                        }},
                    }
                ],
            }
        ],
    }
    plan_path = tmp_path / f"region_plan_{actor_count}.json"
    plan_path.write_text(json.dumps(plan), encoding="utf-8")
    m1_path = _write_m1(tmp_path, endpoint_ids)
    # The caller reads these paths from the test fixture tuple.
    (tmp_path / "_paths.json").write_text(
        json.dumps({"plan": str(plan_path), "timeline": str(timeline_path), "m1": str(m1_path)}),
        encoding="utf-8",
    )
    return plan_path


def _inputs(tmp_path: Path, actor_count: int) -> tuple[Path, Path, Path]:
    plan_path = _write_planned_inputs(tmp_path, actor_count)
    values = json.loads((tmp_path / "_paths.json").read_text(encoding="utf-8"))
    return Path(values["plan"]), Path(values["timeline"]), Path(values["m1"])


@pytest.mark.skipif(
    not BEAGLE_ASSET.is_file() or not BEAGLE_REQUEST.is_file(),
    reason="server Beagle M2 package/request is unavailable",
)
@pytest.mark.parametrize("frame_count", [75, 90, 150])
def test_real_beagle_tracks_support_parameterized_clocks(
    tmp_path: Path, frame_count: int
) -> None:
    plan, timeline, m1 = _inputs(tmp_path, 2)
    output = tmp_path / f"tracks_{frame_count}"
    receipt = materialize_region_actor_tracks(
        region_plan_path=plan,
        planned_timeline_path=timeline,
        room_manifest_path=ROOM_MANIFEST,
        m1_request_path=m1,
        actor_config=_write_actor_config(tmp_path, 2),
        output_directory=output,
        frame_count=frame_count,
        frame_rate_hz=15,
    )

    assert receipt["native_capture"]["status"] == "not_run"
    assert not (output / "frame_records.json").exists()
    assert receipt["clock"]["frame_count"] == frame_count
    load_m1_inputs(ROOM_MANIFEST, output / "m1_capture_request.json")
    for item in receipt["actors"]:
        track = json.loads((output / item["track_path"]).read_text(encoding="utf-8"))
        assert track["native_observed"] is False
        assert len(track["frames"]) == frame_count
        assert all(frame["action_id"] == "walk" for frame in track["frames"])
        assert all(len(frame["joint_targets"]) == 34 for frame in track["frames"])
        assert all(
            frame["native_pending"]["emitter_world_position_m"] is None
            for frame in track["frames"]
        )
        assert track["emitter"]["planned_route_center_is_not_emitter_position"] is True
        assert track["asset"]["actions"]["walk"]["source_action_name"] == "Walking"
        assert track["asset"]["runtime_roles"]["visual"].endswith("visual.glb")


@pytest.mark.skipif(
    not BEAGLE_ASSET.is_file() or not BEAGLE_REQUEST.is_file(),
    reason="server Beagle M2 package/request is unavailable",
)
def test_real_beagle_three_instance_tracks_keep_independent_axes(tmp_path: Path) -> None:
    plan, timeline, m1 = _inputs(tmp_path, 3)
    receipt = materialize_region_actor_tracks(
        region_plan_path=plan,
        planned_timeline_path=timeline,
        room_manifest_path=ROOM_MANIFEST,
        m1_request_path=m1,
        actor_config=_write_actor_config(tmp_path, 3),
        output_directory=tmp_path / "tracks_3",
        frame_count=90,
        frame_rate_hz=15,
    )
    assert len(receipt["actors"]) == 3
    assert len({item["actor_id"] for item in receipt["actors"]}) == 3
    assert len({item["source_endpoint_id"] for item in receipt["actors"]}) == 3


@pytest.mark.skipif(not HUMAN_GLB.is_file(), reason="server human GLB is unavailable")
def test_ue_human_glb_cannot_be_used_as_an_m2_manifest(tmp_path: Path) -> None:
    plan, timeline, m1 = _inputs(tmp_path, 2)
    config = json.loads(_write_actor_config(tmp_path, 2).read_text(encoding="utf-8"))
    config["actors"][0]["asset_manifest_path"] = str(HUMAN_GLB)
    path = tmp_path / "human_as_manifest.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(MP3DRegionActorTrackError, match="usable current M2 Habitat package"):
        materialize_region_actor_tracks(
            region_plan_path=plan,
            planned_timeline_path=timeline,
            room_manifest_path=ROOM_MANIFEST,
            m1_request_path=m1,
            actor_config=path,
            output_directory=tmp_path / "bad",
            frame_count=90,
            frame_rate_hz=15,
        )

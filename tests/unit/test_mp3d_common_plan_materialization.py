from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pytest

from avengine.assets.mp3d_region_actor_tracks import (
    materialize_common_plan_habitat,
)
from avengine.runtime_profiles import (
    build_asset_emitter_binding,
    load_source_asset_runtime_registry,
)


REPOSITORY = Path(__file__).resolve().parents[2]
ROOM_MANIFEST = REPOSITORY / "examples/rooms/habitat_mp3d_example/room_manifest.json"
BASE_M1 = REPOSITORY / "examples/rooms/requests/habitat_mp3d_example.json"
RUNTIME_REGISTRY = REPOSITORY / "examples/runtime/source_asset_runtime_profiles.json"
SPEAKER_A = "generated_bookshelf_speaker_compact_shelf_cabinet_black_ash_research_v1"
SPEAKER_B = "generated_bookshelf_speaker_compact_shelf_cabinet_walnut_veneer_research_v1"
BEAGLE = "rocketbox_dog_beagle_01_m2_v7_world_contact_candidate"
BEAGLE_MANIFEST = Path(
    "/data/avengine_external/datasets/m2/"
    "rocketbox_beagle_m2_canary_v7_world_contact_r5/asset_manifest.json"
)
BEAGLE_REQUEST = Path(
    "/data/avengine_external/review/"
    "current_mp3d_two_beagle_route_lateral_seed22_grounded/primary_m2_request.json"
)


def _clock(frame_count: int = 2) -> dict[str, int | float]:
    return {
        "frame_count": frame_count,
        "frame_rate_hz": 15,
        "sample_rate_hz": 16000,
        "sample_count": round(frame_count * 16000 / 15),
        "time_base_hz": 48000,
        "ticks_per_frame": 3200,
    }


def _camera() -> dict:
    return {
        "position_m": [-8.0, 1.5, -2.0],
        "basis": {
            "forward": [0.0, 0.0, -1.0],
            "right": [1.0, 0.0, 0.0],
            "up": [0.0, 1.0, 0.0],
        },
        "horizontal_fov_deg": 90.0,
    }


def _static_plan(frame_count: int = 2) -> dict:
    registry = load_source_asset_runtime_registry(RUNTIME_REGISTRY)
    actors = []
    for index, asset_id in enumerate((SPEAKER_A, SPEAKER_B), start=1):
        actors.append(
            {
                "actor_id": f"source{index}",
                "source_slot_id": f"source{index}",
                "source_endpoint_id": f"source{index}_emitter",
                "asset_id": asset_id,
                "asset_revision": "sound_source_assets_v1",
                "entity_class": "rigid_object",
                "identity": deepcopy(
                    next(
                        item["identity"]
                        for item in registry["assets"]
                        if item["asset_id"] == asset_id
                    )
                ),
                "realized_attributes": deepcopy(
                    next(
                        item["realized_attributes"]
                        for item in registry["assets"]
                        if item["asset_id"] == asset_id
                    )
                ),
                "timeline": None,
                "emitter_binding": build_asset_emitter_binding(
                    registry,
                    source_slot_id=f"source{index}",
                    asset_id=asset_id,
                ),
                "semantic_id": 210 + index,
            }
        )
    camera = _camera()
    frames = []
    positions = ([-7.0, 0.072447, -3.0], [-9.0, 0.072447, -3.0])
    for frame_index in range(frame_count):
        states = [
            {
                "actor_id": actor["actor_id"],
                "source_slot_id": actor["source_slot_id"],
                "root_transform": {
                    "translation_m": list(positions[index]),
                    "rotation_xyzw": [0.0, 0.0, 0.0, 1.0],
                    "scale": [1.0, 1.0, 1.0],
                },
                "action_id": "static",
                "action_phase": 0.0,
                "action_time_ticks": 0,
                "moving": False,
            }
            for index, actor in enumerate(actors)
        ]
        frames.append(
            {
                "frame_index": frame_index,
                "pts_ticks": frame_index * 3200,
                "actor_states": states,
                "camera_state": deepcopy(camera),
            }
        )
    return {
        "schema": "avengine_common_visual_plan_v1",
        "episode_id": "unit_common_plan",
        "seed": 7,
        "clock": _clock(frame_count),
        "scene": {"room_id": "habitat_mp3d_example_17DRP5sb8fy"},
        "visual_plan": {"camera": camera, "actors": actors, "frames": frames},
    }


@pytest.mark.skipif(
    not RUNTIME_REGISTRY.is_file() or not ROOM_MANIFEST.is_file(),
    reason="repository room/runtime registry unavailable",
)
def test_static_common_plan_preserves_root_camera_clock_and_capture_inputs(
    tmp_path: Path,
) -> None:
    plan = _static_plan()
    plan["seed"] = 202609070501
    receipt = materialize_common_plan_habitat(
        plan=plan,
        room_manifest=ROOM_MANIFEST,
        runtime_registry=RUNTIME_REGISTRY,
        base_m1_request=BASE_M1,
        output=tmp_path / "materialized",
    )
    output = tmp_path / "materialized"
    case = json.loads((output / "case_manifest.json").read_text())
    request = json.loads((output / "m1_capture_request.json").read_text())
    assert receipt["status"] == "research_only"
    assert receipt["checks"]["capture_input_validation"] == "pass"
    assert case["clock"] == plan["clock"]
    assert plan["seed"] == 202609070501
    assert request["seed"] == 202609070501 % (2 ** 31)
    assert receipt["simulator_seed"]["input_seed"] == plan["seed"]
    assert receipt["simulator_seed"]["habitat_seed"] == request["seed"]
    assert [x["source_id"] for x in request["sources"]] == [
        "source1_emitter",
        "source2_emitter",
    ]
    assert request["primary_camera_rig"]["shared_calibration"]["hfov_degrees"] == 90.0
    for index, slot in enumerate(("source1", "source2")):
        track = json.loads((output / f"tracks/{slot}.json").read_text())
        for frame_index, frame in enumerate(track["frames"]):
            expected = plan["visual_plan"]["frames"][frame_index]["actor_states"][index]
            actual = frame["planned_world_from_object"]
            assert np.allclose(
                actual["translation_m"],
                expected["root_transform"]["translation_m"],
                rtol=0.0,
                atol=1.0e-9,
            )
            assert frame["action_id"] == "static"


@pytest.mark.skipif(
    not RUNTIME_REGISTRY.is_file()
    or not ROOM_MANIFEST.is_file()
    or not BEAGLE_MANIFEST.is_file()
    or not BEAGLE_REQUEST.is_file(),
    reason="beagle M2 package or repository inputs unavailable",
)
def test_articulated_common_plan_samples_existing_baked_action_grid(
    tmp_path: Path,
) -> None:
    registry = load_source_asset_runtime_registry(RUNTIME_REGISTRY)
    emitter = build_asset_emitter_binding(
        registry,
        source_slot_id="source1",
        asset_id=BEAGLE,
    )
    actors = []
    for index in (1, 2):
        actors.append(
            {
                "actor_id": f"beagle_{index}",
                "source_slot_id": f"source{index}",
                "source_endpoint_id": f"beagle_{index}_muzzle",
                "asset_id": BEAGLE,
                "asset_revision": "m2_v7_world_contact_r5",
                "entity_class": "articulated_animal",
                "identity": {"species_id": "dog", "breed_id": "beagle"},
                "realized_attributes": {},
                "timeline": {
                    "idle_action_id": "idle",
                    "walking_action_id": "walk",
                },
                "emitter_binding": {
                    **deepcopy(emitter),
                    "source_slot_id": f"source{index}",
                },
                "semantic_id": 210 + index,
            }
        )
    camera = _camera()
    frames = []
    ticks = (0, 3200)
    for frame_index, tick in enumerate(ticks):
        states = [
            {
                "actor_id": actor["actor_id"],
                "source_slot_id": actor["source_slot_id"],
                "root_transform": {
                    "translation_m": [-7.5 - index, 0.402896, -3.0],
                    "rotation_xyzw": [0.0, 0.0, 0.0, 1.0],
                    "scale": [1.0, 1.0, 1.0],
                },
                "action_id": "walk",
                "action_phase": tick / 80000.0,
                "action_time_ticks": tick,
                "moving": True,
            }
            for index, actor in enumerate(actors)
        ]
        frames.append(
            {
                "frame_index": frame_index,
                "pts_ticks": tick,
                "actor_states": states,
                "camera_state": deepcopy(camera),
            }
        )
    plan = {
        "schema": "avengine_common_visual_plan_v1",
        "episode_id": "unit_common_beagle",
        "clock": _clock(2),
        "visual_plan": {"camera": camera, "actors": actors, "frames": frames},
    }
    receipt = materialize_common_plan_habitat(
        plan=plan,
        room_manifest=ROOM_MANIFEST,
        runtime_registry=RUNTIME_REGISTRY,
        base_m1_request=BASE_M1,
        output=tmp_path / "beagle_materialized",
    )
    assert receipt["checks"]["capture_input_validation"] == "pass"
    track = json.loads(
        (tmp_path / "beagle_materialized/tracks/source1.json").read_text()
    )
    assert track["asset"]["runtime_joint_order"]
    assert [frame["action_sample_index"] for frame in track["frames"]] == [0, 1]
    plan["visual_plan"]["frames"][1]["actor_states"][0]["action_phase"] = 0.5
    with pytest.raises(ValueError, match="normalized phase disagrees"):
        materialize_common_plan_habitat(plan=plan, room_manifest=ROOM_MANIFEST,
            runtime_registry=RUNTIME_REGISTRY, base_m1_request=BASE_M1,
            output=tmp_path / "bad_phase")


def test_common_static_track_rejects_hidden_motion(tmp_path):
    plan=_static_plan()
    plan['visual_plan']['frames'][1]['actor_states'][0]['root_transform']['translation_m'][0]+=.2
    with pytest.raises(ValueError,match='root moves despite'):
        materialize_common_plan_habitat(plan=plan,room_manifest=ROOM_MANIFEST,
            runtime_registry=RUNTIME_REGISTRY,base_m1_request=BASE_M1,output=tmp_path/'moving_rigid')


def test_common_camera_preserves_sensor_pose_with_nonzero_base_extrinsic(tmp_path):
    plan=_static_plan();base=json.loads(BASE_M1.read_text())
    base['primary_camera_rig']['shared_calibration']['rig_from_sensor']['translation_m']=[.1,.2,.3]
    materialize_common_plan_habitat(plan=plan,room_manifest=ROOM_MANIFEST,
        runtime_registry=RUNTIME_REGISTRY,base_m1_request=base,output=tmp_path/'sensor_offset')
    result=json.loads((tmp_path/'sensor_offset/m1_capture_request.json').read_text())
    from avengine.assets.mp3d_region_actor_tracks import _transform_matrix
    rig=result['primary_camera_rig'];world=_transform_matrix(rig['world_from_rig'],owner='rig')
    sensor=_transform_matrix(rig['shared_calibration']['rig_from_sensor'],owner='sensor')
    assert (world@sensor)[:3,3]==pytest.approx(plan['visual_plan']['camera']['position_m'])
    assert sensor==pytest.approx(np.eye(4))
    assert result['listener']['rig_from_listener']==rig['shared_calibration']['rig_from_sensor']

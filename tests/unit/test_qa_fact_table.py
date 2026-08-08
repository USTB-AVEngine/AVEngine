from __future__ import annotations

import json
import math
from pathlib import Path

import jsonschema
import numpy as np
import pytest

from avengine.qa.fact_table import (
    QAFactTableError,
    center_frustum_track,
    compile_episode_fact_table,
    listener_local_spherical_track,
)
from avengine.qa.pixel_visibility import compile_pixel_visibility_truth
from avengine.sensor_rig_trajectory import materialize_sensor_rig_trajectory

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SCHEMA_PATH = REPOSITORY_ROOT / "schemas" / "avengine_qa_fact_table_v1.schema.json"

FRAME_COUNT = 75
IDENTITY_WXYZ = [1.0, 0.0, 0.0, 0.0]
SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64


def _registry() -> dict:
    def asset(asset_id: str, species: str, offset: list[float], forward: list[float]):
        return {
            "asset_id": asset_id,
            "revision": "test_v1",
            "display_label": species.title(),
            "entity_class": "articulated_animal",
            "identity": {"species_id": species, "breed_id": f"{species}_breed"},
            "realized_attributes": {
                "size": "medium",
                "body_build": "standard",
                "life_stage": "adult",
                "coat_profile": {"profile_id": f"{species}_coat_v1", "value": "black"},
            },
            "timeline": {"local_anatomical_forward_axis": forward},
            "default_emitter_anchor_id": "muzzle",
            "emitter_anchors": [
                {
                    "anchor_id": "muzzle",
                    "anchor_type": "muzzle",
                    "offset_m": offset,
                    "offset_space": "final_scaled_asset_root",
                }
            ],
            "admission_state": "research",
        }

    return {
        "registry_id": "test_registry",
        "revision": "test_rev",
        "assets": [
            asset("asset_dog", "dog", [0.4, 0.6, 0.0], [1.0, 0.0, 0.0]),
            asset("asset_cat", "cat", [0.3, 0.15, 0.0], [1.0, 0.0, 0.0]),
        ],
    }


def _bank_header() -> dict:
    return {
        "frame_count": FRAME_COUNT,
        "frame_rate_hz": 15,
        "seconds_per_episode": 5.0,
        "source_slots": ["source1", "source2"],
    }


def _bank_episode() -> dict:
    """source1 static facing -Z at (0, 0.5, -2); source2 walks +X at 0.3 m/s."""

    static_root = [0.0, 0.5, -2.0]
    static_emitter = [0.0, 1.1, -2.4]  # offset [0.4,0.6,0] yaw-rotated to -Z
    source1_root = [list(static_root) for _ in range(FRAME_COUNT)]
    source1_emitter = [list(static_emitter) for _ in range(FRAME_COUNT)]

    source2_root = [[1.0 + 0.02 * i, 0.3, 1.0] for i in range(FRAME_COUNT)]
    # offset [0.3,0.15,0] yaw-rotated toward -X (facing -X)
    source2_emitter = [[1.0 + 0.02 * i - 0.3, 0.45, 1.0] for i in range(FRAME_COUNT)]
    return {
        "episode_id": "test_episode_0000",
        "motion_case": "source1_static_source2_moving",
        "source_center_paths_m": {
            "source1": source1_emitter,
            "source2": source2_emitter,
        },
        "source_root_paths_m": {"source1": source1_root, "source2": source2_root},
    }


def _sample_entry() -> dict:
    return {
        "episode_id": "test_episode_0000",
        "asset_ids_by_source_slot": {"source1": "asset_dog", "source2": "asset_cat"},
        "audio": {
            "channel_count": 2,
            "sample_rate_hz": 16000,
            "sample_count": 80000,
            "peak_absolute": 0.05,
            "mixture": {"path": "test_episode_0000__v00.wav", "audio_sha256": SHA_A},
        },
    }


def _dry_variants() -> dict:
    def variant(path: str) -> dict:
        return {
            "variant_index": 0,
            "record": {
                "input": {"path": path, "sha256": SHA_B},
                "linear_gain": 0.1,
            },
        }

    return {"source1": variant("/dry/dog.wav"), "source2": variant("/dry/cat.wav")}


def _anchors() -> list[dict]:
    return [
        {
            "anchor_id": "camera_listener_default",
            "kind": "camera_listener_pose",
            "position_m": [0.0, 1.5, 0.0],
            "yaw_deg": 0.0,
        },
        {
            "anchor_id": "marker_front",
            "kind": "marker",
            "position_m": [0.0, 0.5, -4.0],
            "yaw_deg": None,
        },
    ]


def _compile(**overrides) -> dict:
    arguments = dict(
        bank_header=_bank_header(),
        bank_episode=_bank_episode(),
        listener_position_m=[0.0, 1.5, 0.0],
        listener_orientation_wxyz=IDENTITY_WXYZ,
        sample_entry=_sample_entry(),
        dry_variants_by_slot=_dry_variants(),
        registry=_registry(),
        anchors=_anchors(),
        room={"room_capsule_id": "test_room", "revision": "v1"},
        camera={"hfov_degrees": 105.0, "resolution_hw": [720, 1280]},
        rir_cache_request_identity_sha256=SHA_C,
        provenance_inputs=[
            {"role": "trajectory_bank", "path": "/x/bank.json", "sha256": SHA_B}
        ],
    )
    arguments.update(overrides)
    return compile_episode_fact_table(**arguments)


def _moving_listener_trajectory() -> dict:
    return materialize_sensor_rig_trajectory(
        trajectory_id="qa_dynamic_listener_move_v1",
        program={
            "kind": "WAYPOINT_ROUTE",
            "waypoints": [
                {
                    "frame_index": 0,
                    "position_m": [0.0, 1.5, 0.0],
                    "yaw_deg": 0.0,
                },
                {
                    "frame_index": 74,
                    "position_m": [1.0, 1.5, 0.0],
                    "yaw_deg": 0.0,
                },
            ],
            "interpolation": "LINEAR_POSITION_SHORTEST_YAW",
        },
    )


def _rotating_listener_trajectory() -> dict:
    return materialize_sensor_rig_trajectory(
        trajectory_id="qa_dynamic_listener_rotate_v1",
        program={
            "kind": "ROTATE_IN_PLACE",
            "position_m": [0.0, 1.5, 0.0],
            "start_yaw_deg": 0.0,
            "end_yaw_deg": 90.0,
            "yaw_interpolation": "SHORTEST_ARC",
        },
    )


def _pixel_truth_for_episode(trajectory: dict) -> dict:
    height, width = 8, 10
    normal_masks: list[np.ndarray] = []
    source1_target_masks: list[np.ndarray] = []
    source2_target_masks: list[np.ndarray] = []
    for frame_index in range(FRAME_COUNT):
        normal = np.zeros((height, width), dtype=np.int32)
        source1_target = np.zeros_like(normal)
        source2_target = np.zeros_like(normal)
        source2_target[0:2, 0:2] = 22
        normal[0:2, 0:2] = 22
        if frame_index < 45 or frame_index >= 60:
            source1_target[2:6, 3:8] = 11
        if frame_index < 15 or frame_index >= 60:
            normal[2:6, 3:8] = 11
        elif frame_index < 30:
            normal[2:6, 3:5] = 11
        normal_masks.append(normal)
        source1_target_masks.append(source1_target)
        source2_target_masks.append(source2_target)

    pose_ids = [frame["pose_hash"] for frame in trajectory["frames"]]
    common_context = {
        "renderer_backend": "hermetic_same_renderer_canary",
        "rgb_renderer_backend": "hermetic_same_renderer_canary",
        "camera_contract_id": "qa_pixel_camera_canary_v1",
        "semantic_id_namespace": "qa_pixel_semantic_canary_v1",
        "resolution_hw": [height, width],
        "frame_indices": list(range(FRAME_COUNT)),
        "camera_pose_ids": pose_ids,
    }
    return compile_pixel_visibility_truth(
        normal_semantic_masks=normal_masks,
        target_only_semantic_masks_by_instance={
            "source1": source1_target_masks,
            "source2": source2_target_masks,
        },
        semantic_ids_by_instance={"source1": 11, "source2": 22},
        normal_context={"pass_kind": "modal_scene", **common_context},
        target_only_contexts_by_instance={
            instance_id: {
                "pass_kind": "target_only",
                "target_instance_id": instance_id,
                **common_context,
            }
            for instance_id in ("source1", "source2")
        },
    )


def test_doa_track_follows_native_full_circle_convention() -> None:
    listener = [0.0, 1.0, 0.0]
    sources = [
        [0.0, 1.0, -2.0],  # dead ahead
        [2.0, 1.0, 0.0],  # right
        [-2.0, 1.0, 0.0],  # left
        [0.0, 1.0, 2.0],  # behind
        [0.0, 3.0, -2.0],  # ahead and above
    ]
    track = listener_local_spherical_track(sources, listener, IDENTITY_WXYZ)
    assert track["azimuth_deg"][0] == pytest.approx(0.0, abs=1e-12)
    assert track["azimuth_deg"][1] == pytest.approx(90.0, abs=1e-12)
    assert track["azimuth_deg"][2] == pytest.approx(-90.0, abs=1e-12)
    assert abs(track["azimuth_deg"][3]) == pytest.approx(180.0, abs=1e-12)
    assert track["elevation_deg"][4] == pytest.approx(45.0, abs=1e-9)
    assert track["distance_m"][0] == pytest.approx(2.0, abs=1e-12)
    assert track["distance_m"][4] == pytest.approx(math.sqrt(8.0), abs=1e-12)


def test_doa_track_with_yawed_listener() -> None:
    # Listener yawed 90 degrees (facing -X in the anchor convention).
    half = math.radians(90.0) / 2.0
    quaternion = [math.cos(half), 0.0, math.sin(half), 0.0]
    track = listener_local_spherical_track(
        [[-3.0, 1.0, 0.0], [0.0, 1.0, -3.0]], [0.0, 1.0, 0.0], quaternion
    )
    assert track["azimuth_deg"][0] == pytest.approx(0.0, abs=1e-9)
    assert track["azimuth_deg"][1] == pytest.approx(90.0, abs=1e-9)


def test_doa_rejects_non_unit_quaternion_and_coincident_points() -> None:
    with pytest.raises(QAFactTableError):
        listener_local_spherical_track(
            [[1.0, 0.0, 0.0]], [0.0, 0.0, 0.0], [2.0, 0.0, 0.0, 0.0]
        )
    with pytest.raises(QAFactTableError):
        listener_local_spherical_track(
            [[0.0, 1.0, 0.0]], [0.0, 1.0, 0.0], IDENTITY_WXYZ
        )


def test_compiled_fact_table_validates_against_repository_schema() -> None:
    fact_table = _compile()
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    jsonschema.Draft202012Validator(schema).validate(fact_table)


def test_compiled_fact_table_derivations() -> None:
    fact_table = _compile()
    assert fact_table["schema"] == "avengine_qa_fact_table_v1"
    assert fact_table["time"]["ticks_per_frame"] == 3200
    assert fact_table["listener"]["yaw_deg"] == pytest.approx(0.0, abs=1e-9)
    assert fact_table["listener"]["static"] is True
    assert "positions_m_by_frame" not in fact_table["listener"]

    source1 = fact_table["tracks"]["instances"]["source1"]
    assert all(not moving for moving in source1["moving"])
    assert source1["facing_yaw_deg"][0] == pytest.approx(0.0, abs=1e-9)
    # Static dog emitter at (0, 1.1, -2.4) seen from (0, 1.5, 0): dead ahead.
    assert source1["doa"]["azimuth_deg"][0] == pytest.approx(0.0, abs=1e-9)
    assert source1["doa"]["distance_m"][0] == pytest.approx(
        math.hypot(0.4, 2.4), abs=1e-9
    )

    source2 = fact_table["tracks"]["instances"]["source2"]
    assert all(source2["moving"])
    assert source2["speed_mps"][0] == pytest.approx(0.3, abs=1e-9)
    # Cat muzzle displaced toward -X while walking +X: facing -X is +90 yaw.
    assert source2["facing_yaw_deg"][0] == pytest.approx(90.0, abs=1e-9)

    events = {event["source_slot_id"]: event for event in fact_table["sound_events"]}
    assert events["source1"]["end_tick"] == 240000
    assert events["source1"]["sound_class"]["species_id"] == "dog"
    assert events["source2"]["dry_variant"]["input_path"] == "/dry/cat.wav"

    pair = fact_table["tracks"]["pairwise"]["source1__source2"]
    first_expected = math.dist([0.0, 1.1, -2.4], [0.7, 0.45, 1.0])
    assert pair["emitter_distance_m"][0] == pytest.approx(first_expected, abs=1e-9)

    relations = fact_table["relations"]["anchor_distances"]
    assert "camera_listener_default" not in relations["source1"]
    assert relations["source1"]["marker_front"]["min_m"] == pytest.approx(
        math.dist([0.0, 1.1, -2.4], [0.0, 0.5, -4.0]), abs=1e-9
    )


def test_dynamic_listener_position_updates_per_frame_facts_and_doa() -> None:
    fact_table = _compile(sensor_rig_trajectory=_moving_listener_trajectory())
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    jsonschema.Draft202012Validator(schema).validate(fact_table)

    listener = fact_table["listener"]
    assert listener["static"] is False
    assert (
        listener["sensor_rig_trajectory"]["trajectory_id"]
        == "qa_dynamic_listener_move_v1"
    )
    assert listener["sensor_rig_trajectory"]["dynamic"] is True
    assert len(listener["positions_m_by_frame"]) == FRAME_COUNT
    assert listener["positions_m_by_frame"][0] == pytest.approx([0.0, 1.5, 0.0])
    assert listener["positions_m_by_frame"][-1] == pytest.approx([1.0, 1.5, 0.0])
    assert listener["yaw_deg_by_frame"] == pytest.approx([0.0] * FRAME_COUNT)

    doa = fact_table["tracks"]["instances"]["source1"]["doa"]
    assert doa["azimuth_deg"][0] == pytest.approx(0.0, abs=1e-9)
    assert doa["azimuth_deg"][-1] == pytest.approx(
        math.degrees(math.atan2(-1.0, 2.4)), abs=1e-9
    )
    assert doa["distance_m"][0] == pytest.approx(math.sqrt(0.4**2 + 2.4**2))
    assert doa["distance_m"][-1] == pytest.approx(
        math.sqrt(1.0**2 + 0.4**2 + 2.4**2)
    )


def test_dynamic_listener_orientation_updates_per_frame_doa() -> None:
    fact_table = _compile(sensor_rig_trajectory=_rotating_listener_trajectory())
    listener = fact_table["listener"]
    assert listener["static"] is False
    assert listener["positions_m_by_frame"][0] == pytest.approx(
        listener["positions_m_by_frame"][-1]
    )
    assert listener["yaw_deg_by_frame"][0] == pytest.approx(0.0, abs=1e-9)
    assert listener["yaw_deg_by_frame"][-1] == pytest.approx(90.0, abs=1e-9)
    doa = fact_table["tracks"]["instances"]["source1"]["doa"]
    assert doa["azimuth_deg"][0] == pytest.approx(0.0, abs=1e-9)
    assert doa["azimuth_deg"][-1] == pytest.approx(90.0, abs=1e-9)


def test_sensor_rig_trajectory_rejects_mismatch_and_invalid_frames() -> None:
    trajectory = _moving_listener_trajectory()
    with pytest.raises(QAFactTableError, match="frame 0 position"):
        _compile(
            listener_position_m=[0.1, 1.5, 0.0],
            sensor_rig_trajectory=trajectory,
        )

    invalid = json.loads(json.dumps(trajectory))
    invalid["frames"][1]["pts_ticks"] += 1
    with pytest.raises(QAFactTableError, match="sensor_rig_trajectory is invalid"):
        _compile(sensor_rig_trajectory=invalid)


def test_compiled_fact_table_binds_pixel_visibility_truth() -> None:
    trajectory = _moving_listener_trajectory()
    pixel_truth = _pixel_truth_for_episode(trajectory)
    fact_table = _compile(
        sensor_rig_trajectory=trajectory,
        pixel_visibility_truth=pixel_truth,
        camera={"hfov_degrees": 105.0, "resolution_hw": [8, 10]},
    )
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    jsonschema.Draft202012Validator(schema).validate(fact_table)
    bound = fact_table["visibility"]["pixel_truth"]
    assert bound["status"] == "computed_modal_target_only_v1"
    assert bound["renderer_backend"] == bound["rgb_renderer_backend"]
    assert bound["per_instance"]["source1"]["state_counts"] == {
        "out_of_view": 15,
        "visible_clear": 30,
        "visible_occluded": 15,
        "fully_occluded": 15,
    }
    assert bound["per_instance"]["source2"]["state_counts"] == {
        "out_of_view": 0,
        "visible_clear": 75,
        "visible_occluded": 0,
        "fully_occluded": 0,
    }


def test_pixel_visibility_truth_rejects_sensor_rig_pose_mismatch() -> None:
    trajectory = _moving_listener_trajectory()
    pixel_truth = _pixel_truth_for_episode(trajectory)
    pixel_truth["camera_pose_ids"][10] = "different_camera_pose"
    with pytest.raises(QAFactTableError, match="camera poses differ"):
        _compile(
            sensor_rig_trajectory=trajectory,
            pixel_visibility_truth=pixel_truth,
            camera={"hfov_degrees": 105.0, "resolution_hw": [8, 10]},
        )


def test_human_style_vertical_emitter_offset_yields_null_facing() -> None:
    registry = _registry()
    registry["assets"][0]["emitter_anchors"][0]["offset_m"] = [0.0, 1.61, 0.0]
    registry["assets"][0]["timeline"]["local_anatomical_forward_axis"] = [0.0, 0.0, 1.0]
    episode = _bank_episode()
    episode["source_center_paths_m"]["source1"] = [
        [0.0, 0.5 + 1.61, -2.0] for _ in range(FRAME_COUNT)
    ]
    fact_table = _compile(registry=registry, bank_episode=episode)
    assert fact_table["tracks"]["instances"]["source1"]["facing_yaw_deg"] is None


def test_center_frustum_track_geometry_and_events() -> None:
    # hfov 90 deg and a square sensor: both half-tangents are exactly 1.
    listener = [0.0, 0.0, 0.0]
    points = [
        [-3.0, 0.0, -2.0],  # outside left (|x| > depth)
        [-1.5, 0.0, -2.0],  # inside
        [0.0, 0.0, -2.0],  # inside, dead ahead
        [2.5, 0.0, -2.0],  # outside right
        [0.0, 0.0, 2.0],  # behind
        [0.0, 2.5, -2.0],  # above the vertical frustum
    ]
    track = center_frustum_track(
        points,
        listener,
        IDENTITY_WXYZ,
        hfov_degrees=90.0,
        resolution_hw=(100, 100),
    )
    assert track["in_frustum"] == [False, True, True, False, False, False]
    assert track["in_frustum_frame_count"] == 2
    assert not track["always_outside_frustum"]
    events = track["events"]
    assert [(event["kind"], event["frame"], event["side"]) for event in events] == [
        ("entry", 1, "left"),
        ("exit", 3, "right"),
    ]


def test_compiled_fact_table_reports_center_point_visibility() -> None:
    fact_table = _compile()
    visibility = fact_table["visibility"]
    assert visibility["status"] == "computed_center_point_v0"
    assert visibility["hfov_degrees"] == pytest.approx(105.0)
    # Static dog dead ahead is inside; both instances have full tracks.
    assert visibility["per_instance"]["source1"]["always_inside_frustum"]
    assert len(visibility["per_instance"]["source2"]["in_frustum"]) == FRAME_COUNT
    # Cat sits behind the identity-facing camera (z = +1): never in frustum.
    assert visibility["per_instance"]["source2"]["always_outside_frustum"]
    assert fact_table["frame_events"]["events"] == []


def test_rejects_wrong_frame_count() -> None:
    episode = _bank_episode()
    episode["source_center_paths_m"]["source1"] = episode["source_center_paths_m"][
        "source1"
    ][:-1]
    with pytest.raises(QAFactTableError, match="frame count"):
        _compile(bank_episode=episode)


def test_rejects_emitter_path_that_disagrees_with_registry_offset() -> None:
    episode = _bank_episode()
    episode["source_center_paths_m"]["source1"] = [
        [0.0, 1.2, -2.4] for _ in range(FRAME_COUNT)
    ]
    with pytest.raises(QAFactTableError, match="emitter"):
        _compile(bank_episode=episode)


def test_rejects_unknown_asset_and_missing_dry_variant() -> None:
    sample = _sample_entry()
    sample["asset_ids_by_source_slot"]["source1"] = "asset_missing"
    with pytest.raises(QAFactTableError, match="registry"):
        _compile(sample_entry=sample)
    with pytest.raises(QAFactTableError, match="dry variant"):
        _compile(dry_variants_by_slot={"source1": _dry_variants()["source1"]})


def test_rejects_audio_duration_mismatch_and_bad_cache_identity() -> None:
    sample = _sample_entry()
    sample["audio"]["sample_count"] = 79999
    with pytest.raises(QAFactTableError, match="duration"):
        _compile(sample_entry=sample)
    with pytest.raises(QAFactTableError, match="sha256"):
        _compile(rir_cache_request_identity_sha256="not-a-hash")


def test_rejects_duplicate_anchor_ids() -> None:
    anchors = _anchors() + [_anchors()[1]]
    with pytest.raises(QAFactTableError, match="duplicate anchor"):
        _compile(anchors=anchors)
